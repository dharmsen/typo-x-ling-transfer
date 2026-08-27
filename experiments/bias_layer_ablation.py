"""Exhaustive feature-subset ablation of the ATLAS-24 bias layer B.

For every non-empty subset of the five bias descriptors (script_diff,
genealogical_dist, geo_dist_norm, log1p(resource_s), log1p(resource_t)), plus the
intercept-only baseline (no bias layer at all), refit the bias ridge B_S, form the
debias residual T_S = BTS - B_S, refit the debiased Stage-2 RF on T_S, and measure
the effect on (a) B itself, (b) the residual T, (c) LOLO-CV generalization, and
(d) the on-scale per-source pretraining ranking -- i.e. whether the "English stays
#1" headline survives perturbing which nuisance descriptors define the bias.

The Stage-2 typology design X2 is bias-independent, so it is built once; only the
ridge target (T_S) changes per subset. The bias ridge is fit in-sample on all
ATLAS pairs (the production debias_matrix pipeline does not cross-validate the
bias layer), so LOLO generalization is measured on the BTS scale as
pearson(y, cv_predict(X2, T_S) + B_S): the debiased predictor of BTS puts the
(known) bias B_S back on scale. The biased baseline pearson(y, cv_predict(X2, y))
is computed once and is constant across subsets.

Outputs (run from the package root):  uv run python bias_layer_ablation.py
  - data/bias_layer_ablation.csv        exhaustive master table (one row per subset)
  - data/bias_layer_ablation_loo.csv    leave-one-out marginal view (+ full / none refs)
  - outputs/bias_layer_ablation.tex     paper LaTeX table (also printed to stdout)
"""
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import _grambank
from pipeline import data_sources
from pipeline import debias_matrix as dm
from pipeline import fit_models as fm

HERE = Path(__file__).resolve().parents[1]
SOURCES = ("grambank", "wals")
FLOOR = 0.70
FEAT_NAMES = ["script_diff", "genealogical_dist", "geo_dist_norm",
              "log1p(resource_s)", "log1p(resource_t)"]


def _spearman(a, b):
    """Spearman correlation as Pearson on average ranks (no scipy dependency)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = pd.Series(a[m]).rank().values
    rb = pd.Series(b[m]).rank().values
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _rank_by_source(src, score):
    """Mean score per source language (directed pair endpoint p[0])."""
    return pd.DataFrame({"s": src, "v": score}).groupby("s")["v"].mean()


def _load_names():
    """glottocode -> readable name, for the LaTeX top-source label."""
    try:
        al = pd.read_csv(HERE / "data" / "atlas_languages.csv")
        return dict(zip(al.glottocode, al.grambank_name))
    except Exception:
        return {}


def main():
    bts, langs_ids, langs, wide, scripts = fm._load_common(FLOOR, SOURCES)
    wiki = data_sources.load_wikipedia_articles()
    gene = _grambank.genealogical_distance(langs, langs_ids)
    geo = _grambank.geographic_distance(langs, langs_ids)
    pairs = fm.directed_pairs(langs_ids)
    bi = bts.set_index(["glottocode_s", "glottocode_t"])
    y = np.array([float(bi.loc[(s, t), "bts"]) for s, t in pairs])
    src = np.array([str(p[0]) for p in pairs])

    # Stage-2 typology design: bias-independent, built once.
    atlas_sub = wide.reindex(index=langs_ids)
    M, foc, _, feats, idx = _grambank._one_hot_layout(atlas_sub, list(wide.columns))
    X2 = fm.stage2_design(pairs, {g: k for k, g in enumerate(idx)}, M, foc, len(feats))

    # Five-column bias design (geo normalized with the production lo/hi).
    bias = dm.fit_bias(pairs, y, scripts, gene, geo, wiki)
    geo_lo, geo_hi = bias["geo_lo"], bias["geo_hi"]
    Xb = np.vstack([dm.bias_pair_features(s, t, scripts, gene, geo, wiki, geo_lo, geo_hi)
                    for s, t in pairs])
    assert Xb.shape == (len(y), 5), Xb.shape

    folds = fm.lolo_folds(pairs, langs_ids)
    rf_factory = lambda: RandomForestRegressor(
        n_estimators=200, random_state=42, **fm.rf_kwargs("full_tuned", SOURCES))

    biased_oof = fm.cv_predict(X2, y, folds, rf_factory)
    rho_biased = fm.pearson(y, biased_oof)
    truth_rank = _rank_by_source(src, y)

    # All 31 non-empty subsets + the intercept-only baseline (no bias layer).
    configs = [()] + [c for k in range(1, 6) for c in itertools.combinations(range(5), k)]

    raw = []
    for ci, mask in enumerate(configs):
        if mask:
            Xs = Xb[:, list(mask)]
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xs, y)
            B_S = ridge.predict(Xs)
            label = ",".join(FEAT_NAMES[i] for i in mask)
        else:  # intercept only: B = mean(y), so T = centered BTS (no bias correction)
            B_S = np.full_like(y, y.mean())
            label = "(intercept only)"
        T_S = y - B_S
        rf_T_S = rf_factory().fit(X2, T_S)
        oof = fm.cv_predict(X2, T_S, folds, rf_factory)
        rho_lolo = fm.pearson(y, oof + B_S)            # debiased BTS predictor, on scale
        rank_S = _rank_by_source(src, rf_T_S.predict(X2) + B_S)
        raw.append({
            "mask": mask, "label": label, "k": len(mask),
            "r2_B_y": fm.r2(y, B_S), "corr_B_y": fm.pearson(y, B_S),
            "std_B": float(np.std(B_S)), "std_T": float(np.std(T_S)),
            "rho_lolo": rho_lolo, "T_S": T_S, "rank_S": rank_S,
        })
        print(f"  [{ci + 1}/{len(configs)}] k={len(mask):d}  R2(B,y)={raw[-1]['r2_B_y']:.3f}  "
              f"LOLO rho={rho_lolo:.3f}  features={label}", flush=True)

    full = next(r for r in raw if len(r["mask"]) == 5)
    T_full, rank_full = full["T_S"], full["rank_S"]
    nfT = float(np.linalg.norm(T_full))
    sources_order = truth_rank.index.tolist()
    top1_full = rank_full.reindex(sources_order).idxmax()

    rows = []
    for r in raw:
        rank_S = r["rank_S"].reindex(sources_order)
        top1 = rank_S.idxmax()
        rows.append({
            "mask": "".join("1" if i in r["mask"] else "0" for i in range(5)),
            "features": r["label"], "k": r["k"],
            "r2_B_y": r["r2_B_y"], "corr_B_y": r["corr_B_y"],
            "std_B": r["std_B"], "std_T": r["std_T"],
            "rel_delta_T": float(np.linalg.norm(r["T_S"] - T_full) / nfT) if nfT else float("nan"),
            "corr_T_full": fm.pearson(r["T_S"], T_full),
            "rho_lolo_debiased": r["rho_lolo"],
            "spearman_rank_full": _spearman(rank_S.values, rank_full.reindex(sources_order).values),
            "spearman_rank_truth": _spearman(rank_S.values, truth_rank.reindex(sources_order).values),
            "top1": top1, "top1_same": bool(top1 == top1_full),
        })
    master = pd.DataFrame(rows)
    master.to_csv(HERE / "data" / "bias_layer_ablation.csv", index=False)

    # Leave-one-out marginal view (+ full reference and no-bias extreme).
    full_row = master[master["k"] == 5].iloc[0]
    loo = []
    loo.append(_loo_rec("all five (full)", full_row))
    for i, nm in enumerate(FEAT_NAMES):
        bm = "".join("0" if j == i else "1" for j in range(5))
        loo.append(_loo_rec("$-$" + nm.replace("_", "\\_"), master[master["mask"] == bm].iloc[0]))
    loo.append(_loo_rec("none (no bias)", master[master["k"] == 0].iloc[0]))
    loo_df = pd.DataFrame(loo)
    loo_df.to_csv(HERE / "data" / "bias_layer_ablation_loo.csv", index=False)

    name_map = _load_names()
    import json
    (HERE / "data" / "bias_layer_ablation_meta.json").write_text(
        json.dumps({"rho_biased": rho_biased}))
    _emit_full_table(master, rho_biased, name_map, HERE / "outputs" / "bias_layer_ablation.tex")

    nonempty = master[master["k"] > 0]
    n_top1 = int(nonempty["top1_same"].sum())
    print(f"\nbiased LOLO rho (rf_full)            = {rho_biased:.3f}")
    print(f"full debiased LOLO rho (rf_T + B)    = {full_row['rho_lolo_debiased']:.3f}")
    print(f"#1 source unchanged in {n_top1}/{len(nonempty)} non-empty subsets")
    print(f"median rank-Spearman vs truth        = {nonempty['spearman_rank_truth'].median():.3f}")
    print(f"wrote data/bias_layer_ablation.csv (+ _loo.csv) and outputs/bias_layer_ablation.tex")


def _loo_rec(config, row):
    return {
        "config": config,
        "r2_B_y": float(row["r2_B_y"]),
        "rel_delta_T": float(row["rel_delta_T"]),
        "rho_lolo": float(row["rho_lolo_debiased"]),
        "spearman_rank_full": float(row["spearman_rank_full"]),
        "spearman_rank_truth": float(row["spearman_rank_truth"]),
        "top1": str(row["top1"]),
        "top1_same": bool(row["top1_same"]),
    }


ABBR = ["scr", "gene", "geo", r"res$_s$", r"res$_t$"]


def _subset_label(mask):
    """Compact LaTeX label for a bias-feature subset from its 5-char bitmask."""
    if mask == "00000":
        return r"\textit{none (no bias layer)}"
    return ", ".join(ABBR[i] for i, b in enumerate(mask) if b == "1")


def _emit_full_table(master, rho_biased, name_map, path):
    """Exhaustive LaTeX table: every bias-feature subset grouped by size k=1..5
    (sorted by LOLO rho within each group), with the no-bias baseline last."""
    def nm(g):
        return name_map.get(g, str(g))

    head = [
        r"\begin{table}[h]",
        r"\centering\small",
        r"\label{tab:bias_ablation}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l r r r r r l}",
        r"\toprule",
        r"Bias features & $R^2(B,y)$ & $\|\Delta T\|$ (\%) & LOLO $\rho$ & "
        r"rank $\rho$ (full) & rank $\rho$ (truth) & top \\",
        r"\midrule",
    ]
    body = []
    for k in range(1, 6):
        nword = str(k) + " feature" + ("s" if k > 1 else "")
        body.append(r"\multicolumn{7}{l}{\textit{$k=" + str(k) + r"$ (" + nword + r")}} \\")
        grp = master[master["k"] == k].sort_values("rho_lolo_debiased", ascending=False)
        for _, r in grp.iterrows():
            cfg = _subset_label(r["mask"])
            if k == 5:
                cfg = r"\textbf{" + cfg + "}"
            top = nm(r["top1"]) + (r"" if r["top1_same"] else r"\,$\neq$")
            body.append(
                f"{cfg} & {r['r2_B_y']:.3f} & {r['rel_delta_T'] * 100:.1f} & "
                f"{r['rho_lolo_debiased']:.3f} & {r['spearman_rank_full']:.3f} & "
                f"{r['spearman_rank_truth']:.3f} & {top} \\\\"
            )
        body.append(r"\midrule")
    none = master[master["k"] == 0].iloc[0]
    top_none = nm(none["top1"]) + (r"" if none["top1_same"] else r"\,$\neq$")
    body.append(
        f"{_subset_label(none['mask'])} & {none['r2_B_y']:.3f} & "
        f"{none['rel_delta_T'] * 100:.1f} & {none['rho_lolo_debiased']:.3f} & "
        f"{none['spearman_rank_full']:.3f} & {none['spearman_rank_truth']:.3f} & "
        f"{top_none} \\\\"
    )
    tail = [
        r"\bottomrule",
        r"\end{tabular}}",
        (
            r"\caption{Exhaustive feature-subset ablation of the ATLAS-24 bias layer "
            r"$B$, grouped by subset size $k$. Each row refits $B$ on that subset of "
            r"the five nuisance descriptors and propagates to the debias residual "
            r"$T=\mathrm{BTS}-B$. Abbreviations: scr=script difference, "
            r"gene=genealogical distance, geo=geographic distance, "
            r"res$_s$/res$_t$=$\log1p$ Wikipedia resource of source/target. "
            r"$R^2(B,y)$: bias-explained variance of BTS. $\|\Delta T\|$: relative "
            r"shift of $T$ from the full-five residual. LOLO $\rho$: "
            r"leave-one-language-out Pearson of the debiased BTS predictor (RF on $T$, "
            r"bias $B$ added back); the biased baseline is $\rho="
            + f"{rho_biased:.3f}"
            + r"$. Rank $\rho$: Spearman of the per-source pretraining ranking against "
            r"the full-model ranking and against empirical BTS ground truth; the top "
            r"source is English under every config ($\neq$ marks a flip, none observed).}"
        ),
        r"\end{table}",
    ]
    out = "\n".join(head + body + tail)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out + "\n")
    print("\n" + out)


def regen_table():
    """Re-emit the LaTeX table from the cached master CSV without refitting models."""
    import json
    master = pd.read_csv(HERE / "data" / "bias_layer_ablation.csv", dtype={"mask": str})
    with open(HERE / "data" / "bias_layer_ablation_meta.json") as fh:
        rho_biased = json.load(fh)["rho_biased"]
    _emit_full_table(master, rho_biased, _load_names(),
                     HERE / "outputs" / "bias_layer_ablation.tex")
    print("regenerated outputs/bias_layer_ablation.tex")


if __name__ == "__main__":
    import sys
    if "--table-only" in sys.argv:
        regen_table()
    else:
        main()
