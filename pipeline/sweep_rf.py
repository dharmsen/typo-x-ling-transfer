# sweep_rf.py
"""Exhaustive curated hyperparameter sweep for the Stage-2 RandomForest.

Reuses the tested fit_models pipeline (design matrix + LOLO folds, bit-identical)
and evaluates each RF config under TWO cross-validation protocols:

  1. Repeated LOLO  -- 24 language-folds (the existing protocol); operationally
     relevant because the in-set RF only ever interpolates among the ATLAS-24.
  2. Repeated grouped language K-fold  -- hold out one language group; TEST pairs
     are those whose BOTH languages are held out (pure unseen-language regime);
     train pairs never touch a held-out language, so there is no leakage.

Metric of record = true R^2 (sklearn r2_score) on held-out predictions. Pearson
rho is reported alongside and used only as a tie-break. This replaces the
"R^2 ~ rho^2" hand-wave: for a shrunk RF the two rank configs differently.

Two stages (the one-hot design matrix is config-independent -> built once per
source; the RF fit is the only per-task cost):
  screen : LOLO only, n_estimators=200, single seed, over the curated grid.
  refine : top-K screen configs + the two reference configs, BOTH protocols,
           multi-seed, n_estimators in {200, 500}.

Outputs (run from inside this directory):
  data/rf_sweep_screen__<source>.csv  per-config screen results (LOLO)
  data/rf_sweep_refine__<source>.csv  per-config refine results (both protocols)
  data/rf_sweep_best.csv        per-source combined-rank winner + per-protocol winners
Figures are rendered separately by `plot_sweep.py` (outputs/figures/, gitignored).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from pipeline import _grambank
from pipeline import data_sources
from pipeline import fit_models as fm

HERE = Path(__file__).resolve().parents[1]
BASE_SEED = 42

# Committed `full` (sklearn defaults) LOLO Pearson, per source -- the correctness
# gate: the default reference config must reproduce these to 1e-6, proving the
# sweep reuses fit_models' design + CV bit-identically. (data/d2_model_comparison*.csv)
FULL_LOLO_RHO = {
    ("grambank",): 0.7275994302184003,
    ("wals",): 0.56009976833866869,
    ("grambank", "wals"): 0.615112363559163,
}

# Curated screen grid: the regularization surface that matters at tiny n on a
# wide one-hot design (the `max_features=0.3` fix showed feature subsampling is
# dominant; the rest of this surface has never been searched).
MAX_FEATURES = [0.2, 0.3, 0.5, 0.7, 1.0, "sqrt", "log2"]
MAX_DEPTH = [None, 6, 10, 16]
MIN_SAMPLES_LEAF = [1, 2, 5]
MAX_SAMPLES = [0.6, 1.0]


# --- reference configs -------------------------------------------------------
# `default` == sklearn defaults == the committed `full` tier (the gate target).
# `current_tuned` == {"max_features": 0.3} == the committed `full_tuned` tier.
REFERENCE_CONFIGS = {
    "default": {},
    "current_tuned": {"max_features": 0.3},
}


def screen_configs():
    """Yield rf_params dicts (without n_estimators / random_state) for the grid."""
    for mf, md, msl, msamp in itertools.product(
            MAX_FEATURES, MAX_DEPTH, MIN_SAMPLES_LEAF, MAX_SAMPLES):
        yield {
            "max_features": mf,
            "max_depth": md,
            "min_samples_leaf": msl,
            "max_samples": msamp,
            "bootstrap": True,
        }


# --- metrics -----------------------------------------------------------------
def _agg(values):
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def _r2(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = np.isfinite(pred) & np.isfinite(y)
    if m.sum() < 3 or np.std(y[m]) < 1e-12:
        return float("nan")
    return float(r2_score(y[m], pred[m]))


def lolo_metrics(X2, y, folds, rf_params, n_estimators, seeds):
    """Repeated LOLO: reuse fm.cv_predict (last-write-wins per pair) per seed."""
    r2s, rhos = [], []
    for seed in seeds:
        factory = lambda s=seed: RandomForestRegressor(
            n_estimators=n_estimators, random_state=s, **rf_params)
        pred = fm.cv_predict(X2, y, folds, factory)
        r2s.append(_r2(y, pred))
        rhos.append(fm.pearson(y, pred))
    r2m, r2sd = _agg(r2s)
    rhom, rhosd = _agg(rhos)
    return {"lolo_r2_mean": r2m, "lolo_r2_sd": r2sd,
            "lolo_rho_mean": rhom, "lolo_rho_sd": rhosd}


def holdout_metrics(X2, y, splits, rf_params, n_estimators, seeds):
    """Repeated grouped K-fold: train on both-in-train pairs, predict both-held-out
    pairs. `splits` is a list of (train_idx, test_idx) over X2 row indices.

    Pools all held-out predictions across splits PER SEED and computes one R^2/rho
    over the pool (the standard pooled-CV estimate) -- averaging R^2 across the
    tiny 12-20-pair both-unseen test sets is pure noise (sd ~1.4). The seed-to-seed
    spread then reflects only RF stochasticity, giving a stable, rankable metric.
    """
    r2s, rhos = [], []
    for seed in seeds:
        ys, ps = [], []
        for tr, te in splits:
            if len(te) < 4:  # too few held-out pairs for a meaningful r2
                continue
            m = RandomForestRegressor(
                n_estimators=n_estimators, random_state=seed, **rf_params)
            m.fit(X2[tr], y[tr])
            ps.extend(m.predict(X2[te]).tolist())
            ys.extend(y[te].tolist())
        ys = np.asarray(ys, dtype=float)
        ps = np.asarray(ps, dtype=float)
        r2s.append(_r2(ys, ps))
        rhos.append(fm.pearson(ys, ps))
    r2m, r2sd = _agg(r2s)
    rhom, rhosd = _agg(rhos)
    return {"holdout_r2_mean": r2m, "holdout_r2_sd": r2sd,
            "holdout_rho_mean": rhom, "holdout_rho_sd": rhosd}


# --- CV split construction ---------------------------------------------------
def grouped_kfold_test_lang_sets(langs, k, n_repeats, seed):
    """List of frozenset(test-language-glottocodes) for a repeated grouped K-fold
    over languages (shuffled per repeat)."""
    rng = np.random.default_rng(seed)
    langs = list(langs)
    out = []
    for _ in range(n_repeats):
        perm = rng.permutation(len(langs))
        for group in np.array_split(perm, k):
            out.append(frozenset(langs[i] for i in group))
    return out


def pair_idx(pairs, test_langs):
    """Row indices into X2 for the both-in-test and both-in-train pairs."""
    ts = set(test_langs)
    te = [i for i, (s, t) in enumerate(pairs) if s in ts and t in ts]
    tr = [i for i, (s, t) in enumerate(pairs) if s not in ts and t not in ts]
    return tr, te


# --- per-source prep ---------------------------------------------------------
def prep_source(sources, mode="gb"):
    """Build the config-independent Stage-2 design + targets + LOLO folds once."""
    bts, langs_ids, langs, wide, scripts = fm._load_common(0.70, sources, mode)
    sub = wide.reindex(index=langs_ids)
    M, foc, has, feats, idx = _grambank._one_hot_layout(sub, list(wide.columns))
    pos = {g: k for k, g in enumerate(idx)}
    pairs = fm.directed_pairs(langs_ids)
    bts_idx = bts.set_index(["glottocode_s", "glottocode_t"])
    y = np.array([float(bts_idx.loc[(s, t), "bts"]) for s, t in pairs])
    X2 = fm.stage2_design(pairs, pos, M, foc, len(feats))
    folds = fm.lolo_folds(pairs, langs_ids)
    print(f"[{','.join(sources)}] n_lang={len(langs_ids)} pairs={len(pairs)} "
          f"features={len(feats)} X2_cols={X2.shape[1]}")
    return dict(langs_ids=langs_ids, pairs=pairs, y=y, X2=X2, folds=folds,
                n_lang=len(langs_ids))


def _cfg_row(tag, rf_params, n_estimators, metrics, stage):
    row = {"tag": tag, "stage": stage, "n_estimators": n_estimators}
    for k in ("max_features", "max_depth", "min_samples_leaf",
              "min_samples_split", "max_samples", "bootstrap", "criterion"):
        row[k] = rf_params.get(k, pd.NA)
    row.update(metrics)
    return row


# --- stages ------------------------------------------------------------------
def run_screen(src, n_jobs, screen_seed=BASE_SEED):
    """LOLO-only screen over references + the grid. n_estimators=200, single seed."""
    refs = [("default", REFERENCE_CONFIGS["default"]),
            ("current_tuned", REFERENCE_CONFIGS["current_tuned"])]
    grid = list(screen_configs())

    def work(tag_params):
        tag, params = tag_params
        m = lolo_metrics(src["X2"], src["y"], src["folds"], params,
                         n_estimators=200, seeds=[screen_seed])
        row = _cfg_row(tag, params, 200, m, "screen")
        print(f"    [screen] {tag:>14s} mf={str(params.get('max_features')):>5s} "
              f"md={str(params.get('max_depth')):>4s} loloR2={m['lolo_r2_mean']:.3f}", flush=True)
        return row

    items = refs + [("grid", p) for p in grid]
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(work)(it) for it in items)
    df = pd.DataFrame(rows)
    # tag grid rows uniquely so they survive a concat but keep params in columns
    df.loc[df["tag"] == "grid", "tag"] = [
        f"grid_{i}" for i in range((df["tag"] == "grid").sum())]
    return df.sort_values("lolo_r2_mean", ascending=False).reset_index(drop=True)


def run_refine(src, configs, n_jobs, lolo_seeds, holdout_seeds,
               holdout_k=5, holdout_repeats=5, n_estimators=(200, 500)):
    """Re-evaluate configs under BOTH protocols, multi-seed, at each n_estimators.

    `configs` is a list of (tag, rf_params)."""
    splits = [pair_idx(src["pairs"], ts) for ts in grouped_kfold_test_lang_sets(
        src["langs_ids"], holdout_k, holdout_repeats, BASE_SEED)]

    def work(tag_params, n_est):
        tag, params = tag_params
        out = {"lolo": lolo_metrics(src["X2"], src["y"], src["folds"], params,
                                    n_est, lolo_seeds)}
        out["holdout"] = holdout_metrics(src["X2"], src["y"], splits, params,
                                         n_est, holdout_seeds)
        row = _cfg_row(tag, params, n_est, {}, "refine")
        row.update(out["lolo"])
        row.update(out["holdout"])
        return row

    tasks = [(tp, n) for tp in configs for n in n_estimators]
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(work)(tp, n) for (tp, n) in tasks)
    return pd.DataFrame(rows)


# --- driver ------------------------------------------------------------------
def _coerce_params(params):
    """Coerce hyperparam values read from a CSV row (the grid column is mixed
    float/str -> object dtype, so e.g. 0.5 comes back as '0.5') back to the
    sklearn types RandomForestRegressor expects."""
    out = {}
    for k, v in params.items():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if isinstance(v, str):
            s = v.strip()
            if s.lower() in ("nan", "", "none"):
                continue
            if s in ("sqrt", "log2"):
                out[k] = s
            elif s.lower() in ("true", "false"):
                out[k] = s.lower() == "true"
            elif k in ("max_depth", "min_samples_leaf", "min_samples_split"):
                out[k] = int(round(float(s)))
            else:  # max_features (float), max_samples (float)
                out[k] = float(s)
        else:  # already numeric/bool (in-memory path)
            if k in ("max_depth", "min_samples_leaf", "min_samples_split") and not pd.isna(v):
                out[k] = int(round(float(v)))
            else:
                out[k] = v
    return out


def select_finalists(screen_df, top_k):
    """Top-K screen grid rows by LOLO R^2, deduped by their params, plus both
    reference configs. Returns list of (tag, rf_params)."""
    keep_cols = ["max_features", "max_depth", "min_samples_leaf", "max_samples", "bootstrap"]
    grid = screen_df[screen_df["tag"].str.startswith("grid")].copy()
    finalists = []
    seen = set()
    for _, r in grid.iterrows():
        key = tuple(r[c] for c in keep_cols)
        if key in seen:
            continue
        seen.add(key)
        params = _coerce_params({c: r[c] for c in keep_cols})
        finalists.append((r["tag"], params))
        if len(finalists) >= top_k:
            break
    # always include the two references for an apples-to-apples comparison
    finalists.append(("default", REFERENCE_CONFIGS["default"]))
    finalists.append(("current_tuned", REFERENCE_CONFIGS["current_tuned"]))
    return finalists


def pick_winners(refine_df):
    """Per source: combined-rank winner (rank by LOLO R^2 + rank by holdout R^2,
    lowest sum wins) plus the per-protocol winners. Considers every refine row
    (both n_estimators values) so the data chooses 200 vs 500 trees, not us."""
    d = refine_df.dropna(subset=["lolo_r2_mean", "holdout_r2_mean"]).copy()
    d["rank_lolo"] = d["lolo_r2_mean"].rank(ascending=False)
    d["rank_holdout"] = d["holdout_r2_mean"].rank(ascending=False)
    d["rank_sum"] = d["rank_lolo"] + d["rank_holdout"]
    best_combined = d.sort_values(["rank_sum", "lolo_rho_mean"], ascending=[True, False]).iloc[0]
    best_lolo = d.sort_values(["lolo_r2_mean", "lolo_rho_mean"], ascending=[False, False]).iloc[0]
    best_holdout = d.sort_values(["holdout_r2_mean", "holdout_rho_mean"], ascending=[False, False]).iloc[0]
    return best_combined, best_lolo, best_holdout


def gate(screen_df, sources, mode="gb"):
    """Correctness gate: the `default` reference must reproduce committed `full`.

    gb mode pins the committed n=24 number (FULL_LOLO_RHO). union/intersect have
    no committed number, so read fit_models' `full` for the SAME language set from
    data/d2_model_comparison{suffix}.csv (run `pipeline/fit_models.py --model full
    --lang-set <mode>` first)."""
    if mode == "gb":
        expected = FULL_LOLO_RHO[sources]
    else:
        suffix = data_sources.sources_suffix(sources, mode)
        path = HERE / "data" / f"d2_model_comparison{suffix}.csv"
        if not path.exists():
            raise SystemExit(
                f"gate needs {path.name} -- run `uv run python pipeline/fit_models.py --model "
                f"full --sources {','.join(sources)} --lang-set {mode}` first.")
        expected = float(pd.read_csv(path).set_index("model").loc["full", "lolo_pearson"])
    got = screen_df.loc[screen_df["tag"] == "default", "lolo_rho_mean"].iloc[0]
    ok = abs(got - expected) < 1e-6
    status = "PASS" if ok else "FAIL"
    print(f"  [gate {status}] default LOLO rho={got:.10f} expected={expected:.10f}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="grambank,wals",
                    help="comma-separated CLDF sources (default grambank,wals)")
    ap.add_argument("--stage", choices=["screen", "refine", "both"], default="both")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--n-jobs", type=int, default=14)
    ap.add_argument("--lolo-seeds", type=int, default=5)
    ap.add_argument("--holdout-seeds", type=int, default=3)
    ap.add_argument("--source-sets", default=None,
                    help="explicit ';' list of source-sets to run, each ','-joined "
                         "(default: the three: grambank ; wals ; grambank,wals)")
    ap.add_argument("--lang-set", choices=["gb", "union", "intersect"], default="gb",
                    help="modeling language set (default gb = n=24; union = GB OR WALS "
                         "floor, n=25; intersect = BOTH, n=12). Outputs are mode-suffixed "
                         "so the gb tables are left untouched.")
    a = ap.parse_args()
    mode = a.lang_set

    if a.source_sets:
        sets = [tuple(s.strip() for s in ss.split(",")) for ss in a.source_sets.split(";")]
    else:
        sets = [("grambank",), ("wals",), ("grambank", "wals")]

    all_results = []
    best_rows = []
    for sources in sets:
        suffix = data_sources.sources_suffix(sources, mode)
        print(f"\n=== sources={list(sources)} (suffix={suffix!r}, lang-set={mode}) ===")
        src = prep_source(sources, mode)

        screen_df = None
        if a.stage in ("screen", "both"):
            screen_df = run_screen(src, a.n_jobs)
            if not gate(screen_df, sources, mode):
                raise SystemExit("Correctness gate FAILED -- sweep design is not "
                                 "bit-identical to fit_models; fix before trusting results.")
            screen_df.to_csv(HERE / "data" / f"rf_sweep_screen{suffix}.csv", index=False)
            print(f"  screen top-5 by LOLO R^2:")
            print(screen_df.head(5)[["tag", "max_features", "max_depth",
                                    "min_samples_leaf", "max_samples",
                                    "lolo_r2_mean", "lolo_rho_mean"]].to_string(index=False))

        if a.stage == "refine":
            # need a screen table to pick finalists
            p = HERE / "data" / f"rf_sweep_screen{suffix}.csv"
            screen_df = pd.read_csv(p)

        if a.stage in ("refine", "both"):
            finalists = select_finalists(screen_df, a.top_k)
            print(f"  refining {len(finalists)} configs under both protocols...")
            refine_df = run_refine(
                src, finalists, a.n_jobs,
                lolo_seeds=list(range(BASE_SEED, BASE_SEED + a.lolo_seeds)),
                holdout_seeds=list(range(BASE_SEED, BASE_SEED + a.holdout_seeds)),
                n_estimators=(200,))  # 200 already stable (sd~0.01); 500 doubles
                                      # runtime w/o changing rankings; prod uses 200
            refine_df.to_csv(HERE / "data" / f"rf_sweep_refine{suffix}.csv", index=False)

            combined, blolo, bhold = pick_winners(refine_df)
            for label, row in (("combined", combined), ("lolo", blolo), ("holdout", bhold)):
                print(f"  [{label}] {row['tag']}: LOLO R2={row['lolo_r2_mean']:.3f} "
                      f"holdout R2={row['holdout_r2_mean']:.3f} "
                      f"(mf={row['max_features']} md={row['max_depth']} "
                      f"msl={row['min_samples_leaf']} msamp={row['max_samples']})")
                best_rows.append({
                    "sources": ",".join(sources), "winner": label, "tag": row["tag"],
                    "n_estimators": int(row["n_estimators"]),
                    "max_features": row["max_features"], "max_depth": row["max_depth"],
                    "min_samples_leaf": row["min_samples_leaf"],
                    "max_samples": row["max_samples"], "bootstrap": row["bootstrap"],
                    "lolo_r2_mean": row["lolo_r2_mean"], "lolo_r2_sd": row["lolo_r2_sd"],
                    "lolo_rho_mean": row["lolo_rho_mean"],
                    "holdout_r2_mean": row["holdout_r2_mean"],
                    "holdout_r2_sd": row["holdout_r2_sd"],
                    "holdout_rho_mean": row["holdout_rho_mean"]})

            # merged per-config table for the report
            merged = pd.concat(
                [screen_df.assign(stage="screen"),
                 refine_df.assign(stage="refine")], ignore_index=True, sort=False)
            all_results.append(merged.assign(sources=",".join(sources)))

    agg_suffix = "" if mode == "gb" else "_" + mode
    if all_results:
        pd.concat(all_results, ignore_index=True).to_csv(
            HERE / "data" / f"rf_sweep_all{agg_suffix}.csv", index=False)
    if best_rows:
        pd.DataFrame(best_rows).to_csv(HERE / "data" / f"rf_sweep_best{agg_suffix}.csv", index=False)
        print(f"\nwrote data/rf_sweep_all{agg_suffix}.csv, data/rf_sweep_best{agg_suffix}.csv "
              "(+ per-stage rf_sweep_{screen,refine}__*.csv). Figures: "
              "uv run python plot_sweep.py")


if __name__ == "__main__":
    main()
