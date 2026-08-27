"""Is the E5 residual flip meaningful, or does the operationalization produce
decisive-looking rankings from nothing?

The production debias pipeline fits a ridge B on five nuisance descriptors
in-sample, forms the residual T = BTS - B, fits rf_T on typological features,
and ranks sources by mean in-set rf_T prediction — the ranking whose top is
Marathi/Filipino with English at #11 (Experiment E5's "flip"). Two controls:

  leakfree  The layer explains R^2=0.12 in-sample but ~0 under leave-one-
            language-out CV. Refit B out-of-fold (LOLO: each pair's bias
            predicted by a ridge that never saw its held-out language), form
            the leak-free residual, and recompute the ranking. If the flip is
            carried by the in-sample optimism of B, it vanishes here.

  permnull  Run the ENTIRE operationalization on label-permuted BTS (B=200):
            fit B on permuted y, residualize, fit rf_T, rank sources. If the
            null manufactures top-source margins comparable to the observed
            Marathi margin, the ranking's sharpness carries no evidence of
            meaning.

Both the raw residual mean per source and the rf_T-smoothed ranking (the
production construction) are reported for the observed and leak-free variants.
Run from the package root:  uv run python residual_ranking_null.py
  - data/residual_ranking_gap_test.csv        observed + leak-free ranking summary
  - data/residual_ranking_permutation_null.csv   per-replicate permutation-null records
"""
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)
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
SEED = 42
N_PERM = 200
ENGLISH = "stan1293"
MARA = "mara1378"


def _names():
    al = pd.read_csv(HERE / "data" / "atlas_languages.csv")
    return dict(zip(al.glottocode, al.grambank_name))


def _spearman(a, b):
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def _rank_stats(rank, ref_rank, truth_rank, label, names):
    """One summary row for a per-source mean-residual ranking (descending)."""
    ordered = rank.sort_values(ascending=False)
    top1 = ordered.index[0]
    eng_score = float(rank.get(ENGLISH, np.nan))
    return {
        "variant": label,
        "top1": top1, "top1_name": names.get(top1, top1),
        "top1_score": float(ordered.iloc[0]),
        "english_rank": int(list(ordered.index).index(ENGLISH) + 1),
        "english_score": eng_score,
        "gap_top1_english": float(ordered.iloc[0] - eng_score),
        "margin_over_median": float(ordered.iloc[0] - np.median(rank.values)),
        "spearman_vs_observed": _spearman(rank.reindex(ref_rank.index),
                                          ref_rank.reindex(rank.index)),
        "spearman_vs_truth": _spearman(rank.reindex(truth_rank.index),
                                       truth_rank.reindex(rank.index)),
    }


def main():
    bts, langs_ids, langs, wide, scripts = fm._load_common(FLOOR, SOURCES)
    wiki = data_sources.load_wikipedia_articles()
    gene = _grambank.genealogical_distance(langs, langs_ids)
    geo = _grambank.geographic_distance(langs, langs_ids)
    pairs = fm.directed_pairs(langs_ids)
    bi = bts.set_index(["glottocode_s", "glottocode_t"])
    y = np.array([float(bi.loc[(s, t), "bts"]) for s, t in pairs])
    src = np.array([str(p[0]) for p in pairs])
    names = _names()

    # Stage-2 typology design and five-column bias design, as in production.
    atlas_sub = wide.reindex(index=langs_ids)
    M, foc, _, feats, idx = _grambank._one_hot_layout(atlas_sub, list(wide.columns))
    X2 = fm.stage2_design(pairs, {g: k for k, g in enumerate(idx)}, M, foc, len(feats))
    bias = dm.fit_bias(pairs, y, scripts, gene, geo, wiki)
    geo_lo, geo_hi = bias["geo_lo"], bias["geo_hi"]
    Xb = np.vstack([dm.bias_pair_features(s, t, scripts, gene, geo, wiki, geo_lo, geo_hi)
                    for s, t in pairs])
    folds = fm.lolo_folds(pairs, langs_ids)
    rf_factory = lambda: RandomForestRegressor(
        n_estimators=200, random_state=SEED, **fm.rf_kwargs("full_tuned", SOURCES))

    def bias_fit_predict(Xb_fit, y_fit, Xb_pred):
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xb_fit, y_fit)
        return ridge.predict(Xb_pred)

    def resid_rank(T):
        """Sources ranked by mean in-set rf_T prediction (production E5 ranking)."""
        rf_T = rf_factory().fit(X2, T)
        return pd.DataFrame({"s": src, "v": rf_T.predict(X2)}).groupby("s")["v"].mean()

    def raw_rank(T):
        """Sources ranked by mean residual, no RF smoothing."""
        return pd.DataFrame({"s": src, "v": T}).groupby("s")["v"].mean()

    # --- observed production pipeline (sanity: must reproduce the E5 flip) ---
    B_in = bias_fit_predict(Xb, y, Xb)
    T_in = y - B_in
    rank_obs = resid_rank(T_in)
    truth_rank = pd.DataFrame({"s": src, "v": y}).groupby("s")["v"].mean()
    print(f"observed rf_T ranking: top1={names.get(rank_obs.idxmax(), rank_obs.idxmax())} "
          f"({rank_obs.max():.3f})  English rank="
          f"{list(rank_obs.sort_values(ascending=False).index).index(ENGLISH) + 1} "
          f"({rank_obs.get(ENGLISH):.3f})")
    print(f"(committed E5 values: Marathi 0.361 top, English #11 at 0.125)")

    # --- leak-free bias layer: B refit out-of-fold under the LOLO folds ---
    B_oof = np.zeros_like(y)
    for tr, te in folds:
        B_oof[te] = bias_fit_predict(Xb[tr], y[tr], Xb[te])
    T_oof = y - B_oof
    r2_B_oof = fm.r2(y, B_oof)
    print(f"\nbias layer R^2: in-sample {fm.r2(y, B_in):.3f} -> out-of-fold {r2_B_oof:.3f} "
          f"(pearson {fm.pearson(y, B_oof):.3f})")

    rows = [
        _rank_stats(rank_obs, rank_obs, truth_rank, "observed rf_T (production)", names),
        _rank_stats(raw_rank(T_in), rank_obs, truth_rank, "observed raw mean T", names),
        _rank_stats(resid_rank(T_oof), rank_obs, truth_rank, "leakfree rf_T", names),
        _rank_stats(raw_rank(T_oof), rank_obs, truth_rank, "leakfree raw mean T", names),
    ]
    pd.DataFrame(rows).to_csv(HERE / "data" / "residual_ranking_gap_test.csv", index=False)
    print("\n" + pd.DataFrame(rows).to_string(index=False))

    # --- permutation null: the full operationalization on permuted labels ---
    rng = np.random.default_rng(SEED)
    obs_margin = rows[0]["margin_over_median"]
    obs_gap = rows[0]["gap_top1_english"]
    prec = []
    for b in range(N_PERM):
        y_p = rng.permutation(y)
        B_p = bias_fit_predict(Xb, y_p, Xb)
        T_p = y_p - B_p
        rank_p = resid_rank(T_p)
        ord_p = rank_p.sort_values(ascending=False)
        prec.append({
            "b": b,
            "margin": float(ord_p.iloc[0] - np.median(rank_p.values)),
            "spread": float(ord_p.iloc[0] - ord_p.iloc[-1]),
            "gap_top1_english": float(ord_p.iloc[0] - rank_p.get(ENGLISH, np.nan)),
            "top1": ord_p.index[0],
        })
        if (b + 1) % 25 == 0:
            print(f"  perm replicate {b + 1}/{N_PERM}", flush=True)
    perm = pd.DataFrame(prec)
    perm.to_csv(HERE / "data" / "residual_ranking_permutation_null.csv", index=False)

    p_margin = float((perm.margin >= obs_margin).mean())
    p_gap = float((perm.gap_top1_english >= obs_gap).mean())
    print(f"\npermutation null (B={N_PERM}): margin over median "
          f"{perm.margin.mean():.3f} [{perm.margin.quantile(.025):.3f}, "
          f"{perm.margin.quantile(.975):.3f}] vs observed {obs_margin:.3f}"
          f"  -> p={p_margin:.3f}")
    print(f"top1-vs-English gap: null {perm.gap_top1_english.mean():.3f} "
          f"[{perm.gap_top1_english.quantile(.025):.3f}, "
          f"{perm.gap_top1_english.quantile(.975):.3f}] vs observed {obs_gap:.3f}"
          f"  -> p={p_gap:.3f}")
    n_top1 = perm.top1.value_counts()
    print("null top1 counts (top 5): " +
          ", ".join(f"{names.get(g, g)}:{c}" for g, c in n_top1.head(5).items()))
    print("wrote data/residual_ranking_gap_test.csv and data/residual_ranking_permutation_null.csv")


if __name__ == "__main__":
    main()
