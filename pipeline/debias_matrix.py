# debias_matrix.py
"""Decompose ATLAS-inherited bias from typology signal in the open-universe BTS
matrix, validate the decomposition across bias axes (LOSO/LOFO), and select
which few languages bilingual_transfer/ should measure to validate it.

The D4 matrix (transfer_matrix.py) fits BTS ~= f(typology) on ATLAS-24 and
extrapolates, which amplifies ATLAS's biases (sampling, apparatus, English
centrism) across the universe. Here we fit a ridge B on cheap nuisance
descriptors (script, genealogical distance, geography, Wikipedia resource), form
the residual T = BTS - B, train the Stage-2 RF on T (debiased) and on raw BTS
(biased baseline), and report both + where they disagree.

Bias predictions beyond the ATLAS training set are unvalidated; the residual
debiasing is heuristic and underdetermined at n=24 (see the design spec)."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import _grambank
from pipeline import atlas_languages
from pipeline import data_sources
from pipeline import fit_models as fm
from pipeline import transfer_matrix as tm

HERE = Path(__file__).resolve().parents[1]
CHUNK = 20000


def bias_pair_features(s, t, scripts, gene, geo, wiki, geo_lo, geo_hi):
    """Five nuisance descriptors for the bias layer B(s->t):
    [script_diff, genealogical_distance, geo_distance_normalized,
     log1p(resource_s), log1p(resource_t)]. script_diff is 1.0 when the two
     languages use different writing systems, 0.0 when they share one (polarity
     matches fit_models.baseline_pair_features). Missing script -> 1.0 (differ);
     missing Wikipedia resource -> 0; the genealogical/geographic values come
     from the precomputed DataFrames and require both languages to be present
     there (no fallback)."""
    sa, ta = scripts.get(s), scripts.get(t)
    script_diff = 0.0 if (sa and ta and sa == ta) else 1.0
    g = float(gene.loc[s, t])
    dist = float(geo.loc[s, t])
    dist_n = (dist - geo_lo) / (geo_hi - geo_lo) if geo_hi > geo_lo else 0.0
    rs = float(np.log1p(wiki.get(s, 0)))
    rt = float(np.log1p(wiki.get(t, 0)))
    return np.array([script_diff, g, dist_n, rs, rt], dtype=float)


def fit_bias(pairs, y, scripts, gene, geo, wiki):
    """Fit the bias ridge B on (pairs, y). Returns a dict with the ridge and the
    geographic min/max used to normalize distances (so the same scaling applies
    at predict time)."""
    geo_vals = geo.values[np.isfinite(geo.values)]
    geo_lo = float(np.nanmin(geo_vals)) if geo_vals.size else 0.0
    geo_hi = float(np.nanmax(geo_vals)) if geo_vals.size else 1.0
    X = np.vstack([bias_pair_features(s, t, scripts, gene, geo, wiki, geo_lo, geo_hi)
                   for s, t in pairs])
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)
    return {"ridge": ridge, "geo_lo": geo_lo, "geo_hi": geo_hi}


def bias_layer_table(bundle):
    """The bias ridge B evaluated on the ATLAS-24 training pairs (where its
    metadata are defined). Returns DataFrame: source, target, bts, bts_B,
    residual (bts - bts_B). Universe-wide B is not computed — script/Wikipedia
    metadata are ATLAS-anchored (see design spec)."""
    bias = bundle["bias"]
    scripts, gene, geo, wiki = bundle["scripts"], bundle["gene"], bundle["geo"], bundle["wiki"]
    pairs, y = bundle["pairs"], bundle["y"]
    Xb = np.vstack([bias_pair_features(s, t, scripts, gene, geo, wiki,
                                       bias["geo_lo"], bias["geo_hi"]) for s, t in pairs])
    b = bias["ridge"].predict(Xb)
    return pd.DataFrame({
        "source": [str(s) for s, _ in pairs],
        "target": [str(t) for _, t in pairs],
        "bts": y,
        "bts_B": b,
        "residual": y - b,
    })


def build_debiased_rf(floor=0.70, sources=("grambank", "wals")):
    """Fit two Stage-2 RFs on the ATLAS-24 training set over a SHARED one-hot
    column space: rf_full on raw BTS (the biased D4 baseline) and rf_T on the
    residual T = BTS - B (the debiased model). Also fits the bias ridge B and
    returns everything downstream tasks need (bundle keys listed in the plan)."""
    bts, langs_ids, langs, wide, scripts = fm._load_common(floor, sources)
    wiki = data_sources.load_wikipedia_articles()
    gene = _grambank.genealogical_distance(langs, langs_ids)
    geo = _grambank.geographic_distance(langs, langs_ids)
    feat_cols = list(wide.columns)
    atlas_sub = wide.reindex(index=langs_ids)
    M, foc, has, feats, idx = _grambank._one_hot_layout(atlas_sub, feat_cols)
    col_of, _, n_codes = tm.feature_code_columns(atlas_sub, feat_cols)
    pairs = fm.directed_pairs(langs_ids)
    bi = bts.set_index(["glottocode_s", "glottocode_t"])
    y = np.array([float(bi.loc[(s, t), "bts"]) for s, t in pairs])
    X2 = fm.stage2_design(pairs, {g: k for k, g in enumerate(idx)}, M, foc, len(feats))

    bias = fit_bias(pairs, y, scripts, gene, geo, wiki)
    Xb = np.vstack([bias_pair_features(s, t, scripts, gene, geo, wiki,
                                       bias["geo_lo"], bias["geo_hi"]) for s, t in pairs])
    T = y - bias["ridge"].predict(Xb)

    rf_kwargs = fm.rf_kwargs("full_tuned", sources)
    rf_full = RandomForestRegressor(n_estimators=200, random_state=42, **rf_kwargs).fit(X2, y)
    rf_T = RandomForestRegressor(n_estimators=200, random_state=42, **rf_kwargs).fit(X2, T)
    return {
        "rf_full": rf_full, "rf_T": rf_T, "bias": bias,
        "feat_cols": feat_cols, "feat_of_code": foc, "n_feat": len(feats),
        "col_of": col_of, "n_codes": n_codes, "atlas_ids": set(langs_ids),
        "wide": wide, "scripts": scripts, "langs": langs, "bts": bts,
        "gene": gene, "geo": geo, "wiki": wiki,
        "X2": X2, "y": y, "pairs": pairs, "langs_ids": langs_ids,
    }


def _encode_universe(universe, bundle):
    """Encode the universe against the ATLAS column space once. Returns
    (pairs, pos_univ, M_univ) so multiple RFs can predict on the same encoding
    without re-encoding."""
    wide = bundle["wide"]
    universe = [g for g in universe if g in wide.index]
    pairs = fm.directed_pairs(universe)
    M_univ, ulangs = tm.encode_against_atlas(
        wide.reindex(index=universe), bundle["feat_cols"],
        bundle["col_of"], bundle["n_codes"])
    pos_univ = {g: i for i, g in enumerate(ulangs)}
    return pairs, pos_univ, M_univ


def build_debiased_matrix(universe, bundle):
    """Long-format debiased matrix over `universe`: the biased D4 prediction
    (rf_full), the debiased residual prediction (rf_T), their absolute
    disagreement, extrapolation flag, and affinity confidence. Universe-wide;
    encodes once and predicts per RF (the ridge B layer is reported separately
    on ATLAS pairs)."""
    atlas_ids = bundle["atlas_ids"]
    pairs, pos_univ, M_univ = _encode_universe(universe, bundle)
    foc, n_feat = bundle["feat_of_code"], bundle["n_feat"]
    full_pred = tm._chunked_rf_predict(bundle["rf_full"], pairs, pos_univ, M_univ, foc, n_feat)
    T_pred = tm._chunked_rf_predict(bundle["rf_T"], pairs, pos_univ, M_univ, foc, n_feat)
    affinity = tm.atlas_affinity([s for s, _ in pairs], atlas_ids, bundle["wide"])
    extrap = [bool((s not in atlas_ids) or (t not in atlas_ids)) for s, t in pairs]
    return pd.DataFrame({
        "source": [str(s) for s, _ in pairs],
        "target": [str(t) for _, t in pairs],
        "bts_biased": full_pred,
        "bts_T": T_pred,
        "disagreement": np.abs(full_pred - T_pred),
        "is_extrapolation": extrap,
        "confidence": [float(np.sqrt(max(affinity.get(s, 0.0) * affinity.get(t, 0.0), 0.0)))
                       for s, t in pairs],
    })


def reweight_block_weights(pairs, scripts, family):
    """sample_weight = 1 / |script x genus block|, where a pair's block is the
    unordered (script, WALS genus) pair of its two endpoints. Each block's total
    loss contribution then equals 1 (counterfactual uniform sampling across
    script x subfamily groups). Genus (not major family) is used because
    ATLAS-24 is Indo-European-heavy: major family would lump all IE languages
    into one block and neutralize the reweight."""
    keys = []
    for s, t in pairs:
        bk = tuple(sorted([(scripts.get(s, ""), family.get(s, ("", ""))[1]),
                           (scripts.get(t, ""), family.get(t, ("", ""))[1])]))
        keys.append(bk)
    counts = Counter(keys)
    return np.array([1.0 / counts[k] for k in keys], dtype=float)


def build_reweighted_rf(floor=0.70, sources=("grambank", "wals")):
    """Stage-2 RF on RAW BTS with script x family block-reweighting (the
    'uniform-sampling counterfactual' matrix). Returns a bundle with the same
    column-space keys as build_debiased_rf plus rf_rw."""
    bts, langs_ids, langs, wide, scripts = fm._load_common(floor, sources)
    family = data_sources.wals_family_genus()
    feat_cols = list(wide.columns)
    atlas_sub = wide.reindex(index=langs_ids)
    M, foc, has, feats, idx = _grambank._one_hot_layout(atlas_sub, feat_cols)
    col_of, _, n_codes = tm.feature_code_columns(atlas_sub, feat_cols)
    pairs = fm.directed_pairs(langs_ids)
    bi = bts.set_index(["glottocode_s", "glottocode_t"])
    y = np.array([float(bi.loc[(s, t), "bts"]) for s, t in pairs])
    X2 = fm.stage2_design(pairs, {g: k for k, g in enumerate(idx)}, M, foc, len(feats))
    w = reweight_block_weights(pairs, scripts, family)
    rf_rw = RandomForestRegressor(
        n_estimators=200, random_state=42,
        **fm.rf_kwargs("full_tuned", sources)).fit(X2, y, sample_weight=w)
    return {
        "rf_rw": rf_rw, "feat_cols": feat_cols, "feat_of_code": foc, "n_feat": len(feats),
        "col_of": col_of, "n_codes": n_codes, "atlas_ids": set(langs_ids),
        "wide": wide, "X2": X2, "y": y, "pairs": pairs, "langs_ids": langs_ids,
    }


def grouped_folds(pairs, group_of):
    """{group: (train_idx, test_idx)} where the test fold for group g holds out
    every pair touching a language in g. A cross-group pair appears in the test
    fold of each of its endpoints' groups (intentional, for per-group scoring)."""
    groups = sorted({g for g in group_of.values() if g})
    folds = {}
    all_idx = set(range(len(pairs)))
    for g in groups:
        te = [i for i, p in enumerate(pairs)
              if group_of.get(str(p[0])) == g or group_of.get(str(p[1])) == g]
        tr = sorted(all_idx - set(te))
        folds[g] = (tr, te)
    return folds


def grouped_cv_pearson(X, y, pairs, group_of, factory):
    """Per-group held-out Pearson under leave-one-group-out."""
    folds = grouped_folds(pairs, group_of)
    out = {}
    for g, (tr, te) in folds.items():
        if len(te) < 3:
            continue
        m = factory()
        m.fit(X[tr], y[tr])
        out[g] = float(fm.pearson(y[te], m.predict(X[te])))
    return out


def loso_lofo_report(bundle):
    """Leave-one-script-out and leave-one-family-out Pearson for the biased
    Stage-2 RF (does the typology model generalize across the bias axes?)."""
    rf_kwargs = fm.rf_kwargs("full_tuned")  # sources-default; bundle was built gb-equivalent
    factory = lambda: RandomForestRegressor(n_estimators=200, random_state=42, **rf_kwargs)
    family = data_sources.wals_family_genus()
    scripts = bundle["scripts"]
    langs_ids = bundle["langs_ids"]
    script_of = {g: scripts.get(g) for g in langs_ids if scripts.get(g)}
    family_of = {g: family[g][0] for g in langs_ids if g in family}
    X2, y, pairs = bundle["X2"], bundle["y"], bundle["pairs"]
    return {
        "loso": grouped_cv_pearson(X2, y, pairs, script_of, factory),
        "lofo": grouped_cv_pearson(X2, y, pairs, family_of, factory),
    }


def top_sources_by_positive_coverage(long_df, bts_col, n=5, thresh=0.0):
    """Top-n sources by number of targets with BTS(bts_col) > thresh. Returns a
    list of (source, count)."""
    pos = long_df[long_df[bts_col] > thresh]
    counts = pos.groupby("source")["target"].nunique().sort_values(ascending=False)
    return [(s, int(c)) for s, c in counts.head(n).items()]


def top_sources_by_mean_bts(long_df, bts_col, n=5):
    """Top-n sources by mean BTS(bts_col) over their targets. Commensurable
    across columns of different scales (unlike a >0 threshold), so it is the
    right ranking for the counterfactual comparison."""
    means = long_df.groupby("source")[bts_col].mean().sort_values(ascending=False)
    return [(s, float(m)) for s, m in means.head(n).items()]


def counterfactual_recommendation(biased_df, T_df, rw_df,
                                  biased_col="bts_biased", T_col="bts_T", rw_col="bts_rw", n=5):
    """Side-by-side top pretraining sources under the biased, debiased (T), and
    reweighted matrices, ranked by MEAN BTS per source (commensurable across the
    three columns). The headline: does English-dominance survive debiasing?"""
    b = top_sources_by_mean_bts(biased_df, biased_col, n)
    t = top_sources_by_mean_bts(T_df, T_col, n)
    r = top_sources_by_mean_bts(rw_df, rw_col, n)
    rows = []
    for i in range(n):
        rows.append({
            "rank": i + 1,
            "biased": "%s (%+.3f)" % (b[i][0], b[i][1]) if i < len(b) else "",
            "debiased_T": "%s (%+.3f)" % (t[i][0], t[i][1]) if i < len(t) else "",
            "reweighted": "%s (%+.3f)" % (r[i][0], r[i][1]) if i < len(r) else "",
        })
    return pd.DataFrame(rows)


def counterfactual_recommendation_onscale(deb_df, rw_df, bias_layer_df,
                                          biased_col="bts_biased", T_col="bts_T",
                                          rw_col="bts_rw", n=5):
    """On-scale ATLAS-24 counterfactual ranking: top pretraining sources by mean
    BTS per source under the biased RF (rf_full), the debiased RF PUT BACK ON THE
    BTS SCALE (bts_T + bts_B), the reweighted RF, and digitized ground truth.

    Restricted to ATLAS-24 pairs (where the bias ridge B is defined), so all four
    columns are commensurable. Legacy counterfactual_recommendation ranks the
    off-scale residual bts_T and is unchanged."""
    atlas = deb_df[deb_df["is_extrapolation"] == False]
    rw_alias = "_rw_col_"  # internal alias; avoids merge suffix when rw_col == biased_col
    merged = (atlas[["source", "target", biased_col, T_col]]
              .merge(bias_layer_df[["source", "target", "bts_B", "bts"]],
                     on=["source", "target"], how="inner")
              .merge(rw_df[["source", "target", rw_col]].rename(columns={rw_col: rw_alias}),
                     on=["source", "target"], how="inner"))
    merged["bts_T_onscale"] = merged[T_col] + merged["bts_B"]
    b = top_sources_by_mean_bts(merged, biased_col, n)
    t = top_sources_by_mean_bts(merged, "bts_T_onscale", n)
    r = top_sources_by_mean_bts(merged, rw_alias, n)
    g = top_sources_by_mean_bts(merged, "bts", n)
    rows = []
    for i in range(n):
        rows.append({
            "rank": i + 1,
            "biased": "%s (%+.3f)" % (b[i][0], b[i][1]) if i < len(b) else "",
            "debiased_onscale": "%s (%+.3f)" % (t[i][0], t[i][1]) if i < len(t) else "",
            "reweighted": "%s (%+.3f)" % (r[i][0], r[i][1]) if i < len(r) else "",
            "ground_truth": "%s (%+.3f)" % (g[i][0], g[i][1]) if i < len(g) else "",
        })
    return pd.DataFrame(rows)


def select_validation_targets(K, bundle, sources=("grambank", "wals"), floor=0.70):
    """Rank non-ATLAS languages for targeted BTS measurement by:
      - discrimination: mean |bts_biased - bts_T| over the language's pairs with
        ATLAS-24 (where debiasing changes the prediction most),
      - axis novelty: the language's family is absent from ATLAS-24,
      - feasibility: Grambank coverage >= floor (it must be trainable).
    Greedy selection favors high score while covering distinct novel families.
    Returns a DataFrame sorted by score (top K)."""
    wide = bundle["wide"]
    atlas_ids = bundle["atlas_ids"]
    family = data_sources.wals_family_genus()
    atlas_families = {family[g][0] for g in atlas_ids if g in family}
    cov = atlas_languages.compute_coverage(wide)

    candidates = [g for g in wide.index
                  if g not in atlas_ids
                  and float(cov.get(g, 0.0)) >= floor
                  and g in family]
    if not candidates:
        return pd.DataFrame(columns=["glottocode", "family", "family_novel",
                                     "mean_disagreement", "score"])

    # encode members ONCE against the ATLAS column space; predict only on the
    # candidate->ATLAS pairs (the cells the disagreement score is computed over),
    # not the full N^2 universe, so acquisition stays cheap for a large pool.
    members = list(dict.fromkeys(candidates + sorted(atlas_ids)))
    cap = sorted(atlas_ids)
    M_univ, ulangs = tm.encode_against_atlas(
        wide.reindex(index=members), bundle["feat_cols"],
        bundle["col_of"], bundle["n_codes"])
    pos_univ = {g: i for i, g in enumerate(ulangs)}
    foc, n_feat = bundle["feat_of_code"], bundle["n_feat"]
    pairs_xa = [(g, a) for g in candidates for a in cap]
    pred_full = tm._chunked_rf_predict(bundle["rf_full"], pairs_xa, pos_univ, M_univ, foc, n_feat)
    pred_T = tm._chunked_rf_predict(bundle["rf_T"], pairs_xa, pos_univ, M_univ, foc, n_feat)

    rows = []
    for gi, g in enumerate(candidates):
        lo, hi = gi * len(cap), (gi + 1) * len(cap)
        mean_dis = float(np.mean(np.abs(pred_full[lo:hi] - pred_T[lo:hi])))
        fam = family[g][0]
        novel = fam not in atlas_families
        rows.append({"glottocode": g, "family": fam, "family_novel": novel,
                     "mean_disagreement": mean_dis,
                     "score": mean_dis + (1.0 if novel else 0.0)})

    # greedy: prefer a NEW novel family each step, then fill by raw score
    chosen, seen_fam = [], set()
    pool = sorted(rows, key=lambda r: -r["score"])
    for r in pool:
        if len(chosen) >= K:
            break
        if r["family_novel"] and r["family"] not in seen_fam:
            chosen.append(r); seen_fam.add(r["family"])
    for r in pool:                       # fill remaining slots by raw score
        if len(chosen) >= K:
            break
        if r not in chosen:
            chosen.append(r)
    return pd.DataFrame(chosen).sort_values("score", ascending=False).reset_index(drop=True)


def _preset_set(name):
    if name == "atlas24":
        return set(atlas_languages.atlas_glottocodes(0.70))
    if name == "atlas28":
        return set(atlas_languages.atlas_glottocodes(0.0))
    raise ValueError(f"unknown preset {name!r}")


def _print_summary(deb_df, loso_lofo, counter, targets, label):
    macro = lambda d: float(np.nanmean([v for v in d.values() if np.isfinite(v)])) if d else float("nan")
    print(f"[{label}] debiased matrix: {len(deb_df)} cells | "
          f"mean |biased-T| disagreement {deb_df['disagreement'].mean():.3f}")
    print(f"  LOSO macro Pearson {macro(loso_lofo['loso']):.3f} | "
          f"LOFO macro Pearson {macro(loso_lofo['lofo']):.3f}")
    print("  counterfactual top pretraining (biased / debiased-T / reweighted):")
    for _, r in counter.iterrows():
        print(f"    #{r['rank']}  {r['biased']:<14} {r['debiased_T']:<14} {r['reweighted']}")
    print(f"  targeted measurement ({len(targets)} languages): "
          + ", ".join(targets['glottocode'].astype(str).tolist()[:10]))
    print("NOTE: bias decomposition is heuristic (n=24, 1-decimal BTS); weight the "
          "LOSO/LOFO and disagreement numbers, and validate against bilingual_transfer/ "
          "measurements when they arrive.")


def main():
    """CLI: build the debiased matrix, run bias-axis validation, the
    counterfactual recommendation, and the targeted-measurement plan."""
    import argparse
    ap = argparse.ArgumentParser(
        description="Debiased open-universe BTS matrix + bias-axis validation.")
    ap.add_argument("--sources", default="grambank,wals")
    ap.add_argument("--floor", type=float, default=0.70)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--lang-set", choices=["gb", "union", "intersect"], default="gb")
    ap.add_argument("--preset", choices=["atlas24", "atlas28"], default=None)
    ap.add_argument("--target-k", type=int, default=6,
                    help="how many non-ATLAS languages to suggest measuring")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    sources = tuple(s.strip() for s in a.sources.split(","))
    label = a.label or "debiased"
    out = Path(a.out) / f"debiased__{label}"
    out.mkdir(parents=True, exist_ok=True)

    coverage = tm.per_source_coverage(sources)
    wide = data_sources.combined_wide(sources)
    preset = _preset_set(a.preset) if a.preset else None
    universe = tm.resolve_universe(coverage, list(wide.index), a.lang_set, a.threshold,
                                   preset=preset)
    if len(universe) < 2:
        raise SystemExit(f"universe has {len(universe)} language(s); need >= 2")

    deb = build_debiased_rf(a.floor, sources)
    rwb = build_reweighted_rf(a.floor, sources)
    deb_df = build_debiased_matrix(universe, deb)
    deb_df.to_csv(out / f"debiased_long__{label}.csv", index=False)

    # reweighted prediction over the same universe (rf_rw shares the column space)
    rw_pairs, rw_pos, rw_M = _encode_universe(universe, rwb)
    rw_pred = tm._chunked_rf_predict(rwb["rf_rw"], rw_pairs, rw_pos, rw_M,
                                     rwb["feat_of_code"], rwb["n_feat"])
    rw_df = pd.DataFrame({"source": [str(s) for s, _ in rw_pairs],
                          "target": [str(t) for _, t in rw_pairs], "bts_rw": rw_pred})

    loso_lofo = loso_lofo_report(deb)
    counter = counterfactual_recommendation(
        deb_df, deb_df, rw_df, biased_col="bts_biased", T_col="bts_T", rw_col="bts_rw")

    _macro = lambda d: float(np.nanmean([v for v in d.values() if np.isfinite(v)])) if d else float("nan")
    val_rows = [{"axis": "LOSO", "group": g, "pearson": v} for g, v in loso_lofo["loso"].items()]
    val_rows += [{"axis": "LOFO", "group": g, "pearson": v} for g, v in loso_lofo["lofo"].items()]
    val_rows += [{"axis": "LOSO", "group": "macro", "pearson": _macro(loso_lofo["loso"])},
                 {"axis": "LOFO", "group": "macro", "pearson": _macro(loso_lofo["lofo"])}]
    # bias ablation: does the debiased (T-trained) RF still predict raw BTS?
    val_rows += [
        {"axis": "ablation", "group": "full_rf_in_sample", "pearson": float(fm.pearson(deb["y"], deb["rf_full"].predict(deb["X2"])))},
        {"axis": "ablation", "group": "T_rf_in_sample", "pearson": float(fm.pearson(deb["y"], deb["rf_T"].predict(deb["X2"])))},
    ]
    pd.DataFrame(val_rows).to_csv(out / f"bias_validation__{label}.csv", index=False)
    counter.to_csv(out / f"counterfactual__{label}.csv", index=False)
    onscale = counterfactual_recommendation_onscale(deb_df, rw_df, bias_layer_table(deb))
    onscale.to_csv(out / f"counterfactual_onscale__{label}.csv", index=False)
    bias_layer_table(deb).to_csv(out / f"bias_layer__{label}.csv", index=False)

    targets = select_validation_targets(a.target_k, deb, sources=sources, floor=a.floor)
    targets.to_csv(out / f"validation_targets__{label}.csv", index=False)

    _print_summary(deb_df, loso_lofo, counter, targets, label)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
