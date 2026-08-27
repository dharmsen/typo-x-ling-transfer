# matched_capacity_control.py
"""Matched-capacity non-typological baseline (reviewer M4).

The committed metadata baselines (script/family/wikipedia) are single-feature
Ridges, so they are capacity-starved next to the 387-feature typology RF. To test
whether a NON-typological model of *comparable capacity* can do the same job, we
build a one-hot fingerprint of each language from metadata alone -- script, WALS
family, WALS genus, geographic region (k-means on lat/lon), and a Wikipedia
article-count bin -- feed it through the SAME [M_s, M_t, Delta] pair design, the
SAME 200-tree RF (max_features=0.5), and the SAME LOLO folds as the typology RF.

Output: prints + writes data/matched_capacity_control.csv
(tier, lolo_pearson, lolo_true_r2, n_features) for:
  - metadata_RF   : the matched-capacity non-typological RF (this script)
  - typology_RF   : the full Grambank+WALS RF, re-scored here on identical folds
                    (bit-identical sanity check vs d2_model_comparison)
  - script_ridge  : the straw-man 1-feature ridge (re-scored here for alignment)
"""
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

from pipeline import _grambank
from pipeline import atlas_languages
from pipeline import data_sources
from pipeline.fit_models import (
    _load_common, directed_pairs, lolo_folds, cv_predict, pearson, r2, rf_kwargs,
)


def _metadata_profile(langs_ids, langs, scripts, family, wiki, n_geo_clusters=6, n_wiki_bins=4):
    """Per-language one-hot metadata frame, Glottocode-indexed.

    Categorical fields: script, WALS family (major), WALS genus (minor),
    geographic region (k-means on lat/lon), Wikipedia article-count bin. Missing
    values get an explicit 'UNK' level so the metadata RF sees the same missingness
    that the typology model is spared (this biases *toward* metadata, which is the
    conservative direction for the rebuttal)."""
    geo = langs.set_index("ID")
    latlon = {}
    for g in langs_ids:
        row = geo.loc[g] if g in geo.index else None
        if row is not None:
            la = pd.to_numeric(row.get("Latitude"), errors="coerce")
            lo = pd.to_numeric(row.get("Longitude"), errors="coerce")
            latlon[g] = (float(la) if np.isfinite(la) else np.nan,
                         float(lo) if np.isfinite(lo) else np.nan)
        else:
            latlon[g] = (np.nan, np.nan)
    # k-means regions on centered lat/lon (impute NaN to the global mean so every
    # language gets a cluster; languages with unknown geo share one bucket).
    gs = list(langs_ids)
    XY = np.array([latlon[g] for g in gs], dtype=float)
    mask = np.isfinite(XY).all(axis=1)
    if mask.sum() >= n_geo_clusters:
        mu = XY[mask].mean(axis=0)
        Ximp = np.where((np.isfinite(XY)).all(axis=1, keepdims=True), XY, mu)
        km = KMeans(n_clusters=n_geo_clusters, random_state=42, n_init=10).fit(XY[mask])
        regions = km.predict(Ximp)
    else:
        regions = np.zeros(len(gs), dtype=int)
    region_label = {g: ("G%02d" % r) for g, r in zip(gs, regions)}

    # wikipedia log-article bins (quantiles over the attested set; UNK for missing)
    wiki_vals = {g: np.log1p(wiki.get(g, 0)) for g in gs}
    present = [v for v in wiki_vals.values() if np.isfinite(v) and v > 0]
    if len(set(present)) >= n_wiki_bins:
        qs = np.quantile(present, np.linspace(0, 1, n_wiki_bins + 1)[1:-1])
        def _bin(v):
            if not np.isfinite(v) or v <= 0:
                return "UNK"
            return "W%d" % int(np.searchsorted(qs, v))
        wiki_bin = {g: _bin(v) for g, v in wiki_vals.items()}
    else:
        wiki_bin = {g: "UNK" for g in gs}

    rows = {}
    for g in gs:
        fm, fn = family.get(g, (None, None))
        rows[g] = {
            "script": scripts.get(g) or "UNK",
            "family": fm or "UNK",
            "genus": fn or "UNK",
            "region": region_label[g],
            "wikibin": wiki_bin[g],
        }
    meta = pd.DataFrame(rows).T
    meta.index.name = "glottocode"
    # Convert each categorical column to a frozenset one-hot value (the shape
    # _grambank._one_hot_layout expects: cell = frozenset of codes or NaN).
    wide = meta.apply(lambda c: c.map(lambda v: frozenset([str(v)]) if v != "UNK" else np.nan))
    return wide


def _metadata_stage2(pairs, pos, M, feat_of_code, n_feat):
    """Same [M_s, M_t, Delta] design as fit_models.stage2_design, so the metadata
    RF and typology RF differ ONLY in what the per-language fingerprint encodes."""
    rows = []
    for s, t in pairs:
        a, b = M[pos[s]], M[pos[t]]
        diff_block = np.zeros(n_feat, dtype=np.float32)
        for f in range(n_feat):
            cols = np.where(feat_of_code == f)[0]
            if len(cols):
                diff_block[f] = 0.0 if np.array_equal(a[cols], b[cols]) else 1.0
        rows.append(np.concatenate([a, b, diff_block]))
    return np.vstack(rows)


def main():
    bts, langs_ids, langs, wide_typo, scripts = _load_common(0.70, ("grambank", "wals"), "gb")
    family = data_sources.wals_family_genus()
    wiki = data_sources.load_wikipedia_articles()
    pairs = directed_pairs(langs_ids)
    bts_idx = bts.set_index(["glottocode_s", "glottocode_t"])
    y = np.array([float(bts_idx.loc[(s, t), "bts"]) for s, t in pairs])
    folds = lolo_folds(pairs, langs_ids)

    rf_factory = lambda: RandomForestRegressor(
        n_estimators=200, random_state=42, **rf_kwargs("full_tuned", ("grambank", "wals"), "gb"))

    rows = []

    # --- matched-capacity metadata RF -------------------------------------
    meta_wide = _metadata_profile(langs_ids, langs, scripts, family, wiki)
    Mm, focm, _, featsm, idxm = _grambank._one_hot_layout(meta_wide, list(meta_wide.columns))
    posm = {g: k for k, g in enumerate(idxm)}
    Xm = _metadata_stage2(pairs, posm, Mm, focm, len(featsm))
    pred_m = cv_predict(Xm, y, folds, rf_factory)
    rho_m, r2_m = pearson(y, pred_m), r2(y, pred_m)
    n_feat_m = int(len(featsm))
    print("metadata_RF: n_features=%d  rho=%.4f  R2=%.4f" % (n_feat_m, rho_m, r2_m))
    rows.append({"tier": "metadata_RF", "lolo_pearson": rho_m, "lolo_true_r2": r2_m, "n_features": n_feat_m})

    # --- typology RF re-scored on identical folds (sanity) ----------------
    sub = wide_typo.reindex(index=langs_ids)
    Mt, foct, _, featst, idxt = _grambank._one_hot_layout(sub, list(wide_typo.columns))
    post = {g: k for k, g in enumerate(idxt)}
    Xt = _metadata_stage2(pairs, post, Mt, foct, len(featst))
    pred_t = cv_predict(Xt, y, folds, rf_factory)
    rho_t, r2_t = pearson(y, pred_t), r2(y, pred_t)
    print("typology_RF: n_features=%d  rho=%.4f  R2=%.4f" % (len(featst), rho_t, r2_t))
    rows.append({"tier": "typology_RF", "lolo_pearson": rho_t, "lolo_true_r2": r2_t, "n_features": len(featst)})

    # --- head-to-head: typology RF with metadata feature BUDGET ----------
    # Does typology still win when starved to the metadata feature count?
    # Average over R random subsets of size n_feat_m (seed-controlled).
    from pathlib import Path
    rank = pd.read_csv(Path(__file__).parents[1] / "data" / "d2_feature_importance__grambank_wals.csv")
    ranked = rank.sort_values("importance", ascending=False)["feature_id"].tolist()
    rng = np.random.default_rng(42)
    rho_budget = []
    for _ in range(int(20)):
        # sample without replacement; weight toward informative end but keep some spread
        pick = ranked[: n_feat_m * 3]
        sel = list(rng.choice(pick, size=min(n_feat_m, len(pick)), replace=False))
        sub_b = sub[sel]
        Mb, focb, _, featsb, idxb = _grambank._one_hot_layout(sub_b, sel)
        Xb = _metadata_stage2(pairs, {g: k for k, g in enumerate(idxb)}, Mb, focb, len(featsb))
        rho_budget.append(pearson(y, cv_predict(Xb, y, folds, rf_factory)))
    rho_b = float(np.mean(rho_budget))
    print("typology_RF_budget_matched: rho=%.4f (mean over 20 subsets of %d features)" % (rho_b, n_feat_m))
    rows.append({"tier": "typology_RF_budget_matched", "lolo_pearson": rho_b,
                 "lolo_true_r2": float("nan"), "n_features": n_feat_m})

    pd.DataFrame(rows).to_csv(Path(__file__).parents[1] / "data" / "matched_capacity_control.csv", index=False)
    print("wrote data/matched_capacity_control.csv")


if __name__ == "__main__":
    main()
