# fit_models.py
"""Stage 1 (GLM) + Stage 2 (vector) models, LOLO CV, permutation importance,
and the --model {proxy|glm|full|staged} CLI."""
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import _grambank
from pipeline import atlas_languages


def directed_pairs(langs):
    """All ordered (s,t), s != t. Returns list of 2-element object arrays."""
    return [np.array([s, t], dtype=object) for s in langs for t in langs if s != t]


def lolo_folds(pairs, langs):
    """Leave-one-language-out: fold for lang k holds out every pair with k.
    Returns list of (train_idx, test_idx) over `pairs`."""
    folds = []
    for held in langs:
        te = [i for i, p in enumerate(pairs) if held in p]
        tr = [i for i, p in enumerate(pairs) if held not in p]
        folds.append((tr, te))
    return folds


def pair_features_stage1(s, t, sim, gene, geo, scripts):
    """Four interpretable pair-level distances."""
    return np.array([
        float(sim.loc[s, t]),
        0.0 if scripts.get(s) == scripts.get(t) and scripts.get(s) else 1.0,
        float(gene.loc[s, t]),
        float(geo.loc[s, t]),
    ])


def baseline_pair_features(s, t, scripts, family, wiki):
    """All five baseline pair features in a FIXED column order:
    [share_script, share_major, share_minor, log1p(articles_s), log1p(articles_t)].
    Tiers select the columns they need: script=[0], family=[1,2],
    wikipedia=[3,4], combined=[0,1,2,3,4]. Distances: 0.0 = same, 1.0 = differ;
    a missing signal (e.g. a language not in the WALS family map) counts as
    'differs' (1.0), so restricted tiers never see a missing value."""
    ss, ts = scripts.get(s), scripts.get(t)
    share_script = 0.0 if (ss and ss == ts) else 1.0
    fm, fn = family.get(s), family.get(t)
    share_major = 0.0 if (fm and fn and fm[0] == fn[0]) else 1.0
    share_minor = 0.0 if (fm and fn and fm[1] == fn[1]) else 1.0
    log_s = float(np.log1p(wiki.get(s, 0)))
    log_t = float(np.log1p(wiki.get(t, 0)))
    return np.array([share_script, share_major, share_minor, log_s, log_t], dtype=float)


def stage1_design(pairs, sim, gene, geo, scripts):
    return np.vstack([pair_features_stage1(s, t, sim, gene, geo, scripts) for s, t in pairs])


def cv_predict(X_all, y_all, folds, model_factory):
    """Generic LOLO CV: refit per fold, collect held-out predictions."""
    y_all = np.asarray(y_all, dtype=np.float64)
    pred = np.full(len(y_all), np.nan)
    for tr, te in folds:
        m = model_factory()
        m.fit(X_all[tr], y_all[tr])
        # last-write-wins: each pair is predicted in two folds (one per language in it); the fold holding out the alphabetically-later language wins (relevant for future leakage audits).
        pred[te] = m.predict(X_all[te])
    return pred


def pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def r2(y_true, y_pred):
    """True coefficient of determination 1 - SS_res/SS_tot on held-out
    predictions. Unlike pearson**2, this penalizes a miscalibrated predictor: a
    shrunk Ridge/RF can rank pairs well (high rho) yet explain little variance
    because its predictions have compressed spread (slope < 1). true R^2 is the
    default metric; pearson is kept alongside as a secondary column."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3 or np.std(y_true[m]) < 1e-12:
        return float("nan")
    return float(r2_score(y_true[m], y_pred[m]))


def rf_kwargs(model, sources=("grambank", "wals"), mode="gb"):
    """RandomForest hyperparameters for the Stage-2 RF, by model tier / source /
    language set.

    `full_tuned` applies the sweep-optimal max_features (selected by sweep_rf.py
    as the combined-rank winner over LOLO and held-out-language CV, true R^2). The
    optimal subsampling scales with n -- tiny n wants aggressive subsampling:
      - Grambank / Grambank+WALS (n=24), union (n=25) -> 0.5
      - WALS-only (n=22), intersect (n=12)            -> 'log2'  (~11 cols/split)
    This generalizes the prior "max_features=1.0 overfits" note: the smaller the
    modeling set, the more feature subsampling helps. The baseline tiers (`full`,
    `full25`) keep the default max_features ({}) so their committed numbers are
    unchanged. `n_jobs=-1` is always included for parallelism (a speed knob only;
    same `random_state` -> bit-identical results regardless of n_jobs).
    Shared by fit_models and predict_transfer (predict_transfer stays on the gb
    default) so the Stage-2 RF config lives in one place."""
    if model != "full_tuned":
        return {"n_jobs": -1}
    if tuple(sources) == ("wals",) or mode == "intersect":
        return {"max_features": "log2", "n_jobs": -1}
    return {"max_features": 0.5, "n_jobs": -1}


def stage2_design(pairs, pos, M, feat_of_code, n_feat):
    """Directed-pair design: [M[s], M[t], per-feature disagreement block].
    M[s]/M[t] are one-hot rows (length n_codes); the disagreement block is
    length n_feat (1.0 if the two langs differ on that feature's code set, else 0.0).
    Returns a numeric (no-NaN) design matrix."""
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


def permutation_importance(X, y, folds, feat_of_code, n_feat, model_factory, seed=42):
    """Per-feature importance = drop in LOLO-CV Pearson when that feature's
    one-hot columns are shuffled within each fold's training set. `X` must have
    been built by stage2_design (first 2*len(feat_of_code) cols are the M[s]/M[t]
    one-hot blocks). Returns array of length n_feat."""
    rng = np.random.default_rng(seed)
    base = pearson(y, cv_predict(X, y, folds, model_factory))
    cols_by_feat = [list(np.where(feat_of_code == f)[0]) for f in range(n_feat)]
    n_codes = len(feat_of_code)
    imp = np.zeros(n_feat)
    for f in range(n_feat):
        if not cols_by_feat[f]:
            continue
        Xp = X.copy()
        for tr, _ in folds:
            for col in cols_by_feat[f]:
                Xp[tr, col] = Xp[tr, col][rng.permutation(len(tr))]
                Xp[tr, n_codes + col] = Xp[tr, n_codes + col][rng.permutation(len(tr))]
            # also ablate feature f's disagreement column (stage2_design index
            # 2*n_codes + f), else ridge can still exploit f through it and bias
            # importance toward zero.
            Xp[tr, 2 * n_codes + f] = Xp[tr, 2 * n_codes + f][rng.permutation(len(tr))]
        imp[f] = base - pearson(y, cv_predict(Xp, y, folds, model_factory))
    return imp


def _print_source_coverage(wide, langs_ids, sources):
    """Print per-source feature count + mean coverage over the ATLAS modeling set."""
    from pipeline import data_sources
    sub = wide.reindex(index=langs_ids)

    def _cols_for(src):
        pfx = data_sources.SOURCES[src]["prefix"]
        if pfx:
            return [c for c in wide.columns if c.startswith(pfx)]
        claimed = tuple(p["prefix"] for p in data_sources.SOURCES.values() if p["prefix"])
        return [c for c in wide.columns if not c.startswith(claimed)]

    print(f"feature sources {list(sources)} over n={len(langs_ids)} ATLAS langs:")
    for src in sources:
        cols = _cols_for(src)
        if cols:
            mean_cov = float((sub[cols].notna().sum(axis=1) / len(cols)).mean())
            print(f"  {src:10s}: {len(cols):4d} features, mean coverage {mean_cov:.2f}")


def _load_common(floor=0.70, sources=("grambank", "wals"), mode="gb"):
    from pathlib import Path
    from pipeline import data_sources
    here = Path(__file__).parents[1]
    bts = pd.read_csv(here / "data" / "blt_ground_truth.csv")
    bts = bts.dropna(subset=["bts", "glottocode_s", "glottocode_t"])
    keep = set(atlas_languages.modeling_glottocodes(floor, sources, mode))
    wide = data_sources.combined_wide(sources)               # FEATURES (enriched)
    keep &= set(wide.index)  # only model ATLAS-floor langs the sources cover
                             # (no-op for Grambank / Grambank+WALS, which cover
                             # all 24; WALS alone misses ~2 Glottocode mappings)
    bts = bts[bts["glottocode_s"].isin(keep) & bts["glottocode_t"].isin(keep)]
    langs_ids = sorted(set(bts["glottocode_s"]) | set(bts["glottocode_t"]))
    _, _, langs, _ = _grambank.load_data()                   # language metadata (geo/gene)
    sc = pd.read_csv(here / "data" / "scripts.csv")
    scripts = dict(zip(sc["glottocode"], sc["script"]))
    return bts, langs_ids, langs, wide, scripts


def top_feature_ids(n, rank_path=None, sources=("grambank", "wals"), mode="gb", floor=0.70):
    """Top-n feature IDs by permutation importance, read from the source-matched
    ranking (data/d2_feature_importance[__sources][_mode][_floor].csv). Shared by
    --top-features and predict_transfer's `full25` model."""
    from pathlib import Path
    from pipeline import data_sources
    if rank_path is None:
        suffix = data_sources.sources_suffix(sources, mode, floor)
        rank_path = Path(__file__).parents[1] / "data" / f"d2_feature_importance{suffix}.csv"
    if not rank_path.exists():
        raise SystemExit(
            "%s not found - run `uv run python pipeline/fit_models.py --model staged "
            "--sources %s` first to generate the ranking."
            % (rank_path, ",".join(sources)))
    rank = pd.read_csv(rank_path)
    return rank.sort_values("importance", ascending=False)["feature_id"].head(n).tolist()


def _run_baselines(bts, langs_ids, scripts):
    """Four non-typological baseline tiers (script/family/wikipedia/combined),
    each scored under the same LOLO CV as the typology tiers, with within-fold
    feature standardization. Family uses WALS Family+Genus (covers 22 of the 24
    ATLAS langs); script and wikipedia cover all 24. Each tier restricts the
    24-language universe to its own coverage and rebuilds pairs/folds/y there.
    Returns {name: (lolo_pearson, lolo_r2)}."""
    from pipeline import data_sources
    family = data_sources.wals_family_genus()       # {glottocode: (major, minor)}
    wiki = data_sources.load_wikipedia_articles()   # {glottocode: int}
    bts_idx = bts.set_index(["glottocode_s", "glottocode_t"])
    tiers = [
        ("script",    [g for g in langs_ids if g in scripts],                       [0]),
        ("family",    [g for g in langs_ids if g in family],                        [1, 2]),
        ("wikipedia", [g for g in langs_ids if g in wiki],                          [3, 4]),
        ("combined",  [g for g in langs_ids if g in scripts and g in family and g in wiki],
                     [0, 1, 2, 3, 4]),
    ]
    factory = lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    out = {}
    for name, lang_subset, cols in tiers:
        pairs = directed_pairs(lang_subset)
        y = np.array([float(bts_idx.loc[(s, t), "bts"]) for s, t in pairs])
        folds = lolo_folds(pairs, lang_subset)
        X = np.vstack([baseline_pair_features(s, t, scripts, family, wiki)[cols]
                       for s, t in pairs])
        pred = cv_predict(X, y, folds, factory)
        out[name] = (pearson(y, pred), r2(y, pred))
        print("  baseline %s: n=%d R2=%.4f rho=%.4f" % (
            name, len(lang_subset), out[name][1], out[name][0]))
    return out


def run(model="staged", floor=0.70, feat_subset=None, sources=("grambank", "wals"), mode="gb"):
    from pathlib import Path
    from pipeline import data_sources
    here = Path(__file__).parents[1]
    suffix = data_sources.sources_suffix(sources, mode, floor)
    typo_sources = tuple(s for s in sources if s in data_sources.SOURCES)
    run_baselines = "metadata" in sources
    results = {}
    r2s = {}  # parallel to results: model -> true LOLO R^2 (1 - SS_res/SS_tot)
    if typo_sources:
        bts, langs_ids, langs, wide, scripts = _load_common(floor, typo_sources, mode)
        _print_source_coverage(wide, langs_ids, typo_sources)
        sim = _grambank.simple_similarity(_grambank.subset_wide(wide, langs_ids))
        gene = _grambank.genealogical_distance(langs, langs_ids)
        geo = _grambank.geographic_distance(langs, langs_ids)
        pairs = directed_pairs(langs_ids)
        bts_idx = bts.set_index(["glottocode_s", "glottocode_t"])
        y = np.array([float(bts_idx.loc[(s, t), "bts"]) for s, t in pairs])
        folds = lolo_folds(pairs, langs_ids)
        if model in ("proxy", "staged"):
            Xp = stage1_design(pairs, sim, gene, geo, scripts)[:, [0]]  # sim only
            pred = cv_predict(Xp, y, folds, lambda: Ridge(alpha=1.0))
            results["proxy"] = pearson(y, pred); r2s["proxy"] = r2(y, pred)
        if model in ("glm", "staged"):
            X1 = stage1_design(pairs, sim, gene, geo, scripts)
            pred = cv_predict(X1, y, folds, lambda: Ridge(alpha=1.0))
            results["glm"] = pearson(y, pred); r2s["glm"] = r2(y, pred)
        if model in ("full", "full_tuned", "staged"):
            sub = wide.reindex(index=langs_ids)
            rf_factory = lambda: RandomForestRegressor(n_estimators=200, random_state=42)
            rf_factory_tuned = lambda: RandomForestRegressor(
                n_estimators=200, random_state=42, **rf_kwargs("full_tuned", typo_sources, mode))
            M, foc, has, feats, idx = _grambank._one_hot_layout(sub, list(wide.columns))
            pos = {g: k for k, g in enumerate(idx)}
            X2 = stage2_design(pairs, pos, M, foc, len(feats))
            if model in ("full", "staged"):
                pred = cv_predict(X2, y, folds, rf_factory)
                results["full"] = pearson(y, pred); r2s["full"] = r2(y, pred)
            if model in ("full_tuned", "staged"):
                pred = cv_predict(X2, y, folds, rf_factory_tuned)
                results["full_tuned"] = pearson(y, pred); r2s["full_tuned"] = r2(y, pred)
        if model in ("full", "staged"):
            # D2 feature ranking via ridge permutation importance (fast; defensible at n=24)
            imp = permutation_importance(X2, y, folds, foc, len(feats), lambda: Ridge(alpha=1.0))
            pd.DataFrame({"feature_id": feats, "importance": imp}).sort_values(
                "importance", ascending=False).to_csv(
                here / "data" / f"d2_feature_importance{suffix}.csv", index=False)
            if feat_subset is not None:
                # Ablation: rerun the SAME RF on a feature subset (e.g. the top-N features
                # by permutation importance), head-to-head with the all-features baseline
                # above (identical folds and seed). Written to its own table so the
                # proxy/glm/full headline in d2_model_comparison.csv is left untouched.
                Ms, focs, _, featss, idxs = _grambank._one_hot_layout(sub, feat_subset)
                X2s = stage2_design(pairs, {g: k for k, g in enumerate(idxs)}, Ms, focs, len(featss))
                sub_pred = cv_predict(X2s, y, folds, rf_factory)
                sub_r = pearson(y, sub_pred); sub_r2 = r2(y, sub_pred)
                key = "full_top%d" % len(featss)
                print("feature subset (%d features): %s" % (len(featss), ", ".join(featss)))
                pd.DataFrame([
                    {"model": "full", "n_features": len(feats),
                     "lolo_r2": r2s["full"], "lolo_pearson": results["full"]},
                    {"model": key, "n_features": len(featss),
                     "lolo_r2": sub_r2, "lolo_pearson": sub_r},
                ]).to_csv(here / "data" / f"d2_feature_subset{suffix}.csv", index=False)
                results[key] = sub_r; r2s[key] = sub_r2
    if run_baselines:
        # baselines use the SAME modeling set as the typology tiers (gb default
        # = the GB-24; union/intersect = that mode's language set) + scripts, so
        # every tier is compared at the same n.
        if "grambank" in typo_sources:
            b_bts, b_langs, b_scripts = bts, langs_ids, scripts
        else:
            b_bts, b_langs, _, _, b_scripts = _load_common(floor, ("grambank", "wals"), mode)
        for name, (rho, r2v) in _run_baselines(b_bts, b_langs, b_scripts).items():
            results[name] = rho; r2s[name] = r2v
    if feat_subset is None:
        # true R^2 is the primary column; pearson kept for the sweep gate + legacy.
        pd.DataFrame([{"model": k, "lolo_r2": r2s.get(k, float("nan")),
                       "lolo_pearson": v} for k, v in results.items()]).to_csv(
            here / "data" / f"d2_model_comparison{suffix}.csv", index=False)
    if typo_sources:
        print("model comparison (LOLO, n=%d langs) -- R^2 = true 1-SSres/SStot:" % len(langs_ids))
        for k in results:
            print("  %-12s R2=%+.3f  rho=%+.3f" % (k, r2s.get(k, float("nan")), results[k]))
    else:
        print("baseline comparison (LOLO) -- R^2 = true 1-SSres/SStot:")
        for k in results:
            print("  %-12s R2=%+.3f  rho=%+.3f" % (k, r2s.get(k, float("nan")), results[k]))
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["proxy", "glm", "full", "full_tuned", "staged"], default="staged")
    ap.add_argument("--top-features", type=int, default=None,
                    help="also evaluate the RF restricted to the top-N features by "
                         "permutation importance (reads data/d2_feature_importance[__sources].csv); "
                         "writes data/d2_feature_subset[__sources].csv")
    ap.add_argument("--sources", default="grambank,wals",
                    help="comma-separated CLDF sources plus the optional 'metadata' "
                         "baseline source (default: grambank,wals; use 'grambank' for the "
                         "Grambank-only baseline; add 'metadata' to also run the "
                         "script/family/wikipedia baselines, e.g. 'grambank,metadata')")
    ap.add_argument("--lang-set", choices=["gb", "union", "intersect"], default="gb",
                    help="modeling language set: 'gb' (default) = the ATLAS langs clearing "
                         "Grambank's coverage floor (n=24); 'union' = clears the floor in GB "
                         "OR WALS (n=25); 'intersect' = clears it in BOTH (n=12). Result "
                         "tables are mode-suffixed so the n=24 tables are left untouched.")
    ap.add_argument("--floor", type=float, default=0.70,
                    help="per-language coverage floor (default 0.70 -> n=24 ATLAS langs). "
                         "Lowering admits sparser-attested languages (0.50 -> n=26 adds Telugu "
                         "+ Thai; 0.0 -> all 28). Result tables are floor-suffixed (_fNN) when "
                         "not 0.70, so the default tables are left untouched.")
    a = ap.parse_args()
    sources = tuple(s.strip() for s in a.sources.split(","))
    feat_subset = top_feature_ids(a.top_features, sources=sources, mode=a.lang_set, floor=a.floor) if a.top_features is not None else None
    run(a.model, feat_subset=feat_subset, sources=sources, mode=a.lang_set, floor=a.floor)
