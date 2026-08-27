"""Robustness controls for the paper.

Five analyses, each addressing a specific reviewer attack surface. All reuse the
committed helpers (fit_models, experiments, matched_capacity_control) so the
protocols match the paper exactly:

  tune-metadata  The matched-capacity metadata RF borrowed the typology winner's
                 max_features=0.5 and was never tuned itself, so the headline
                 increment (0.705 vs 0.581) conflates features with tuning. Run
                 the SAME curated sweep grid over the metadata RF, refine the top
                 8 with five seeds, and add leave-2-languages-out for the winner
                 (fills the L2LO "--" cells of Table 1). If the tuned metadata RF
                 stays well below the typology RF, the increment is safe.

  composition    LOSO/LOFO macro-rho (0.776/0.729) is compared against pooled
                 LOLO (0.705), but the grouped held-out sets are dominated by
                 cross-script pairs, where every model is stronger (0.728 vs
                 0.600). Composition-matched control: restrict the POOLED LOLO
                 predictions to each group's exact held-out pairs and
                 macro-average. Single-language groups must then coincide with
                 their LOSO fold (internal check).

  digitize       The ground truth is digitized at one decimal, so every BTS cell
                 carries up to +-0.05 rounding error. Add U(-0.05, 0.05) noise to
                 the targets, rerun the full tuned-RF LOLO, and report the rho
                 distribution. RF seed fixed at 42 so only label noise varies.

  shared22       Grambank scores 0.780 at n=24 but WALS 0.648 at n=22: different
                 language sets. Refit all three feature sources on the 22
                 languages the WALS mapping covers, each with its own
                 max_features mini-sweep, identical folds.

  langboot       The committed 95% CI [0.51, 0.82] resamples source languages
                 only. Language-level bootstrap: resample the 24 languages with
                 replacement and weight each pair by the product of its
                 endpoint multiplicities, so both-endpoint dependence enters.

Run from the package root:  uv run python experiments/robustness_controls.py [section ...]
"""
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)
import itertools
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

from pipeline import _grambank
from pipeline import atlas_languages
from pipeline import data_sources
from pipeline import debias_matrix as dm
from pipeline import fit_models as fm
from experiments import run_experiments as el
from experiments.matched_capacity_control import _metadata_profile, _metadata_stage2

SEED = 42
OUT = "data/robustness__%s.csv"

# the curated sweep grid of sweep_rf.py (screening stage), mirrored exactly
MAX_FEATURES = [0.2, 0.3, 0.5, 0.7, 1.0, "sqrt", "log2"]
MAX_DEPTH = [None, 6, 10, 16]
MIN_SAMPLES_LEAF = [1, 2, 5]
MAX_SAMPLES = [0.6, 1.0]


def _typology_design(sources=("grambank", "wals")):
    """Wide matrix, [M_s, M_t, Delta] design, pairs, y, folds for a source set."""
    s = el.load_setup(sources=sources)
    wide = s["wide"].reindex(index=s["langs_ids"])
    M, foc, _, feats, idx = _grambank._one_hot_layout(wide, list(wide.columns))
    pos = {g: k for k, g in enumerate(idx)}
    X = fm.stage2_design(s["pairs"], pos, M, foc, len(feats))
    return dict(s=s, X=X, feats=feats, wide=wide)


def _metadata_design():
    """The matched-capacity metadata fingerprint through the same pair design."""
    bts, langs_ids, langs, _, scripts = fm._load_common(0.70, ("grambank", "wals"), "gb")
    family = data_sources.wals_family_genus()
    wiki = data_sources.load_wikipedia_articles()
    pairs = fm.directed_pairs(langs_ids)
    bts_idx = bts.set_index(["glottocode_s", "glottocode_t"])
    y = np.array([float(bts_idx.loc[(s, t), "bts"]) for s, t in pairs])
    folds = fm.lolo_folds(pairs, langs_ids)
    meta_wide = _metadata_profile(langs_ids, langs, scripts, family, wiki)
    Mm, focm, _, featsm, idxm = _grambank._one_hot_layout(meta_wide, list(meta_wide.columns))
    posm = {g: k for k, g in enumerate(idxm)}
    Xm = _metadata_stage2(pairs, posm, Mm, focm, len(featsm))
    return langs_ids, pairs, y, folds, Xm, len(featsm)


# ---------------------------------------------------------------------------
def tune_metadata():
    print("=" * 78)
    print("TUNED METADATA CONTROL — same curated grid as the typology sweep")
    print("=" * 78)
    langs_ids, pairs, y, folds, Xm, n_feat = _metadata_design()
    print(f"metadata design: {Xm.shape[0]} pairs x {Xm.shape[1]} cols ({n_feat} fields)")

    def run_cfg(params, seed=SEED):
        pred = fm.cv_predict(Xm, y, folds, lambda: RandomForestRegressor(
            n_estimators=200, random_state=seed, n_jobs=1,
            **params, bootstrap=True))
        return fm.pearson(y, pred), fm.r2(y, pred)

    grid = list(itertools.product(MAX_FEATURES, MAX_DEPTH, MIN_SAMPLES_LEAF, MAX_SAMPLES))
    rows_ref = [
        {"tag": "sklearn_default", **{}},
        {"tag": "production_mf0.5"},
    ]
    res = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_cfg)({"max_features": mf, "max_depth": md,
                          "min_samples_leaf": msl, "max_samples": msamp}, SEED)
        for mf, md, msl, msamp in grid)
    screen = pd.DataFrame(
        [{"max_features": mf, "max_depth": md, "min_samples_leaf": msl,
          "max_samples": msamp, "lolo_rho": r[0], "lolo_r2": r[1]}
         for (mf, md, msl, msamp), r in zip(grid, res)])
    screen = screen.sort_values("lolo_r2", ascending=False).reset_index(drop=True)
    print("\nscreening (single seed 42), top 10 by true R^2:")
    print(screen.head(10).to_string(index=False, float_format=lambda v: "%.4f" % v))

    # defaults + production config for reference
    d_rho, d_r2 = run_cfg({})
    p_rho, p_r2 = run_cfg({"max_features": 0.5})
    print(f"\nsklearn-default metadata RF: rho={d_rho:.4f} R2={d_r2:.4f}")
    print(f"production  mf=0.5        : rho={p_rho:.4f} R2={p_r2:.4f}")

    # refine top 8 across five seeds (guard against a single-seed fluke)
    cols = ["max_features", "max_depth", "min_samples_leaf", "max_samples"]

    def _clean(r):
        r = dict(r)
        md = r["max_depth"]
        r["max_depth"] = None if pd.isna(md) else int(md)
        r["min_samples_leaf"] = int(r["min_samples_leaf"])
        return r

    top8 = [_clean(r) for r in screen.head(8)[cols].to_dict("records")]
    refine = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_cfg)(dict(r), seed)
        for r in top8 for seed in (42, 43, 44, 45, 46))
    refine = pd.DataFrame(
        [{**r, "seed": seed, "lolo_rho": v[0], "lolo_r2": v[1]}
         for r, seed, v in zip([r for r in top8 for _ in range(5)],
                               [seed for _ in range(8) for seed in (42, 43, 44, 45, 46)],
                               refine)])
    agg = (refine.groupby(["max_features", "max_depth", "min_samples_leaf", "max_samples"])
           .agg(rho_mean=("lolo_rho", "mean"), rho_sd=("lolo_rho", "std"),
                r2_mean=("lolo_r2", "mean"), r2_sd=("lolo_r2", "std"))
           .reset_index().sort_values("rho_mean", ascending=False))
    print("\nrefinement of the top 8 over seeds 42-46:")
    print(agg.to_string(index=False, float_format=lambda v: "%.4f" % v))
    best = agg.iloc[0]
    print(f"\nBEST tuned metadata RF: rho={best.rho_mean:.4f}+-{best.rho_sd:.4f} "
          f"(mf={best.max_features}, md={best.max_depth}, msl={best.min_samples_leaf}, "
          f"msamp={best.max_samples})")
    screen.to_csv(OUT % "metadata_tune_screen", index=False)
    agg.to_csv(OUT % "metadata_tune_refine", index=False)

    # leave-2-languages-out for the tuned winner + the single-feature ridges
    print("\nleave-2-languages-out (50 partitions) for the tuned metadata winner ...")
    from sklearn.linear_model import Ridge
    src_of = np.array([p[0] for p in pairs])
    tgt_of = np.array([p[1] for p in pairs])
    rng = np.random.default_rng(SEED)
    n = len(langs_ids)

    def l2lo(predict_fn):
        rhos = []
        for _ in range(50):
            perm = rng.permutation(n)
            groups = [tuple(langs_ids[i] for i in perm[g:g + 2]) for g in range(0, n, 2)]
            folds_k = []
            for grp in groups:
                in_grp = np.isin(src_of, list(grp)) | np.isin(tgt_of, list(grp))
                folds_k.append((np.where(~in_grp)[0], np.where(in_grp)[0]))
            rhos.append(fm.pearson(y, predict_fn(folds_k)))
        return float(np.mean(rhos)), float(np.std(rhos))

    best_params = {"max_features": best.max_features,
                   "max_depth": None if pd.isna(best.max_depth) else int(best.max_depth),
                   "min_samples_leaf": int(best.min_samples_leaf),
                   "max_samples": best.max_samples}
    rho_m, sd_m = l2lo(lambda fk: fm.cv_predict(
        Xm, y, fk, lambda: RandomForestRegressor(
            n_estimators=200, random_state=SEED, n_jobs=-1, **best_params)))
    print(f"tuned metadata RF  L2LO rho = {rho_m:.4f} (sd over partitions {sd_m:.4f})")

    # single-feature ridges on the same partitions: [share_script], family
    # [share_major, share_minor], wikipedia [log_s, log_t]
    fam = data_sources.wals_family_genus()
    wiki = data_sources.load_wikipedia_articles()
    sc = dict(zip(pd.read_csv("data/scripts.csv")["glottocode"],
                  pd.read_csv("data/scripts.csv")["script"]))
    Xb = np.vstack([fm.baseline_pair_features(s, t, sc, fam, wiki) for s, t in pairs])
    ridge_rows = []
    for name, cols in (("script", [0]), ("family", [1, 2]), ("wikipedia", [3, 4])):
        r, sd = l2lo(lambda fk, c=cols: fm.cv_predict(
            Xb[:, c], y, fk, lambda: Ridge(alpha=1.0)))
        ridge_rows.append({"tier": name, "l2lo_rho": r, "l2lo_sd": sd})
        print(f"{name:<10} ridge  L2LO rho = {r:.4f} (sd {sd:.4f})")
    pd.DataFrame(ridge_rows + [{"tier": "metadata_RF_tuned",
                                "l2lo_rho": rho_m, "l2lo_sd": sd_m}]
                 ).to_csv(OUT % "metadata_l2lo", index=False)


# ---------------------------------------------------------------------------
def composition():
    print("=" * 78)
    print("COMPOSITION-MATCHED GROUPED HOLDOUT — LOSO/LOFO vs pooled LOLO on the same pairs")
    print("=" * 78)
    s = el.load_setup(sources=("grambank", "wals"))
    wide = s["wide"].reindex(index=s["langs_ids"])
    M, foc, _, feats, idx = _grambank._one_hot_layout(wide, list(wide.columns))
    pos = {g: k for k, g in enumerate(idx)}
    X = fm.stage2_design(s["pairs"], pos, M, foc, len(feats))
    y, pairs, langs_ids, scripts = s["y"], s["pairs"], s["langs_ids"], s["scripts"]
    factory = lambda: RandomForestRegressor(
        n_estimators=200, random_state=SEED, **fm.rf_kwargs("full_tuned", ("grambank", "wals")))

    lolo_pred = fm.cv_predict(X, y, s["folds"], factory)
    family = data_sources.wals_family_genus()
    rows = []
    for axis, group_of in (
            ("LOSO", {g: scripts[g] for g in langs_ids if scripts.get(g)}),
            ("LOFO", {g: family[g][0] for g in langs_ids if g in family})):
        per_group = dm.grouped_cv_pearson(X, y, pairs, group_of, factory)
        for g, v in per_group.items():
            members = [k for k, gg in group_of.items() if gg == g]
            mask = np.array([any(l in members for l in p) for p in pairs])
            rows.append({"axis": axis, "group": g, "n_langs": len(members),
                         "n_pairs": int(mask.sum()),
                         "grouped_rho": v,                      # retrained, group held out
                         "lolo_same_pairs_rho": fm.pearson(y[mask], lolo_pred[mask])})
        df = pd.DataFrame([r for r in rows if r["axis"] == axis])
        macro_g = df.grouped_rho.mean()
        macro_l = df.lolo_same_pairs_rho.mean()
        w_g = np.average(df.grouped_rho, weights=df.n_pairs)
        w_l = np.average(df.lolo_same_pairs_rho, weights=df.n_pairs)
        rows += [{"axis": axis, "group": "macro (unweighted)", "n_langs": "",
                  "n_pairs": int(df.n_pairs.sum()), "grouped_rho": macro_g,
                  "lolo_same_pairs_rho": macro_l},
                 {"axis": axis, "group": "pair-weighted", "n_langs": "",
                  "n_pairs": int(df.n_pairs.sum()), "grouped_rho": w_g,
                  "lolo_same_pairs_rho": w_l}]
        print(f"\n{axis}: grouped macro={macro_g:.4f}  pooled-LOLO-on-same-pairs macro={macro_l:.4f}"
              f"   (pair-weighted: {w_g:.4f} vs {w_l:.4f})")
        print(df.to_string(index=False, float_format=lambda v: "%.4f" % v))
    pd.DataFrame(rows).to_csv(OUT % "composition", index=False)


# ---------------------------------------------------------------------------
def digitize(B=100):
    print("=" * 78)
    print(f"DIGITIZATION-NOISE ROBUSTNESS — U(-0.05, +0.05) on BTS, B={B}")
    print("=" * 78)
    s = el.load_setup(sources=("grambank", "wals"))
    wide = s["wide"].reindex(index=s["langs_ids"])
    M, foc, _, feats, idx = _grambank._one_hot_layout(wide, list(wide.columns))
    pos = {g: k for k, g in enumerate(idx)}
    X = fm.stage2_design(s["pairs"], pos, M, foc, len(feats))
    y, folds = s["y"], s["folds"]
    factory = lambda: RandomForestRegressor(
        n_estimators=200, random_state=SEED, **fm.rf_kwargs("full_tuned", ("grambank", "wals")))
    rng = np.random.default_rng(SEED)
    rho0 = fm.pearson(y, fm.cv_predict(X, y, folds, factory))
    rhos, r2s = [], []
    for b in range(B):
        yb = y + rng.uniform(-0.05, 0.05, size=y.shape)
        pred = fm.cv_predict(X, yb, folds, factory)
        rhos.append(fm.pearson(y, pred))       # scored against the CLEAN target
        r2s.append(fm.r2(y, pred))
        if (b + 1) % 10 == 0:
            print(f"  replicate {b+1}/{B}: mean rho so far {np.mean(rhos):.4f}")
    rhos, r2s = np.array(rhos), np.array(r2s)
    out = pd.DataFrame({"metric": ["lolo_rho", "lolo_r2"],
                        "clean": [rho0, float("nan")],
                        "noise_mean": [rhos.mean(), r2s.mean()],
                        "noise_sd": [rhos.std(ddof=1), r2s.std(ddof=1)],
                        "p2.5": [np.percentile(rhos, 2.5), np.percentile(r2s, 2.5)],
                        "p97.5": [np.percentile(rhos, 97.5), np.percentile(r2s, 97.5)],
                        "min": [rhos.min(), r2s.min()], "B": [B, B]})
    print(out.to_string(index=False, float_format=lambda v: "%.4f" % v))
    # analytic attenuation for reference: rho* = rho / sqrt(1 + sn^2/sy^2) if the
    # target is noisy but predictions are scored on the noisy y
    sn, sy = 0.05 / np.sqrt(3), y.std(ddof=1)
    print(f"(analytic check: noise sd {sn:.4f} vs BTS sd {sy:.4f}; "
          f"attenuation factor {1/np.sqrt(1 + (sn/sy)**2):.4f})")
    out.to_csv(OUT % "digitize", index=False)


# ---------------------------------------------------------------------------
def shared22():
    print("=" * 78)
    print("FEATURE-SOURCE COMPARISON ON THE 22 SHARED WALS LANGUAGES")
    print("=" * 78)
    s22 = el.load_setup(sources=("wals",))            # WALS floor set -> n=22
    langs22 = list(s22["langs_ids"])
    print(f"shared languages: {len(langs22)}")
    pairs = fm.directed_pairs(langs22)
    bts = pd.read_csv("data/blt_ground_truth.csv").dropna(subset=["bts"])
    bts = bts[bts["glottocode_s"].isin(langs22) & bts["glottocode_t"].isin(langs22)]
    bts_idx = bts.set_index(["glottocode_s", "glottocode_t"])
    y = np.array([float(bts_idx.loc[(s, t), "bts"]) for s, t in pairs])
    folds = fm.lolo_folds(pairs, langs22)

    rows = []
    for sources in (("grambank",), ("wals",), ("grambank", "wals")):
        s = el.load_setup(sources=sources)
        wide = s["wide"].reindex(index=langs22)
        M, foc, _, feats, idx = _grambank._one_hot_layout(wide, list(wide.columns))
        pos = {g: k for k, g in enumerate(idx)}
        X = fm.stage2_design(pairs, pos, M, foc, len(feats))

        def run(mf):
            pred = fm.cv_predict(X, y, folds, lambda: RandomForestRegressor(
                n_estimators=200, random_state=SEED, n_jobs=-1, max_features=mf))
            return fm.pearson(y, pred), fm.r2(y, pred)

        sweep = Parallel(n_jobs=-1, backend="loky")(delayed(run)(mf) for mf in MAX_FEATURES)
        best_i = int(np.argmax([r[1] for r in sweep]))   # metric of record: true R^2
        mf_b, (rho_b, r2_b) = MAX_FEATURES[best_i], sweep[best_i]
        # the per-source production config (rf_kwargs full_tuned); its own sweep row
        prod_mf = fm.rf_kwargs("full_tuned", sources).get("max_features")
        rho_prod = dict(zip(map(str, MAX_FEATURES), [r[0] for r in sweep]))[str(prod_mf)] \
            if prod_mf is not None else run(None)[0]
        rows.append({"sources": "+".join(sources), "n_features": len(feats),
                     "best_max_features": str(mf_b), "lolo_rho": rho_b, "lolo_r2": r2_b,
                     "production_max_features_rho": rho_prod})
        print(f"{'+'.join(sources):<16} ({len(feats):>3} feats): best mf={str(mf_b):>5} "
              f"rho={rho_b:.4f} R2={r2_b:.4f}   (production-config rho={rho_prod:.4f})")
        for mf, (r, r2v) in zip(MAX_FEATURES, sweep):
            rows.append({"sources": "+".join(sources), "n_features": len(feats),
                         "best_max_features": str(mf), "lolo_rho": r, "lolo_r2": r2v,
                         "production_max_features_rho": np.nan})
    pd.DataFrame(rows).to_csv(OUT % "shared22", index=False)


# ---------------------------------------------------------------------------
def _weighted_pearson(a, b, w):
    m = np.isfinite(a) & np.isfinite(b)
    a, b, w = a[m], b[m], w[m]
    wa, wb = np.average(a, weights=w), np.average(b, weights=w)
    cov = np.sum(w * (a - wa) * (b - wb)) / np.sum(w)
    sa = np.sqrt(np.sum(w * (a - wa) ** 2) / np.sum(w))
    sb = np.sqrt(np.sum(w * (b - wb) ** 2) / np.sum(w))
    return float(cov / (sa * sb)) if sa > 0 and sb > 0 else float("nan")


def langboot(B=2000):
    print("=" * 78)
    print(f"LANGUAGE-LEVEL BOOTSTRAP CI — resample languages, weight pairs by w_s * w_t, B={B}")
    print("=" * 78)
    s = el.load_setup(sources=("grambank", "wals"))
    wide = s["wide"].reindex(index=s["langs_ids"])
    M, foc, _, feats, idx = _grambank._one_hot_layout(wide, list(wide.columns))
    pos = {g: k for k, g in enumerate(idx)}
    X = fm.stage2_design(s["pairs"], pos, M, foc, len(feats))
    y, pairs, langs_ids = s["y"], s["pairs"], s["langs_ids"]
    factory = lambda: RandomForestRegressor(
        n_estimators=200, random_state=SEED, **fm.rf_kwargs("full_tuned", ("grambank", "wals")))
    pred = fm.cv_predict(X, y, s["folds"], factory)
    src = np.array([p[0] for p in pairs])
    tgt = np.array([p[1] for p in pairs])
    rng = np.random.default_rng(SEED)
    n = len(langs_ids)
    rhos = []
    for _ in range(B):
        draw = rng.integers(0, n, size=n)
        w = np.bincount(draw, minlength=n).astype(float)
        wmap = dict(zip(range(n), w))
        pw = np.array([wmap[langs_ids.index(a)] * wmap[langs_ids.index(b)]
                       for a, b in zip(src, tgt)])
        rhos.append(_weighted_pearson(y, pred, pw))
    rhos = np.array(rhos)
    out = pd.DataFrame({"metric": ["lolo_rho_langboot"],
                        "estimate": [fm.pearson(y, pred)],
                        "ci_lo": [np.percentile(rhos, 2.5)],
                        "ci_hi": [np.percentile(rhos, 97.5)], "B": [B]})
    print(out.to_string(index=False, float_format=lambda v: "%.4f" % v))
    out.to_csv(OUT % "langboot", index=False)


SECTIONS = {"tune-metadata": tune_metadata, "composition": composition,
            "digitize": digitize, "shared22": shared22, "langboot": langboot}

if __name__ == "__main__":
    args = sys.argv[1:] or list(SECTIONS)
    for a in args:
        SECTIONS[a]()
