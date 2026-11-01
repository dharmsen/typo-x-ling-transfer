#!/usr/bin/env python3
# nested_tuning.py
"""Nested hyperparameter selection for the Stage-2 RandomForest.

sweep_rf.py selects the production winner by LOLO + grouped-holdout R^2 computed
on the SAME 24 languages that the reported scores are evaluated on, so the
reported numbers carry selection optimism. This module quantifies it:

  for each outer LOLO fold i:
    1. select a winner by INNER cross-validation computed entirely on fold i's
       training pairs (inner LOLO over the 23 remaining languages + inner
       grouped both-unseen holdout), combined rank-sum criterion as in
       production -- but applied to every grid config (no top-8 screen cut);
    2. fit that winner on fold i's training pairs and predict fold i's test
       pairs.

The aggregated outer predictions are the nested score of the whole selection
procedure; the drop against the committed global-selection numbers is the
selection-optimism estimate the paper reports (Appendix, matched-tuning
control: nested typology rho 0.580 against its reported 0.706).

Interrupt/resume: every completed work unit -- one (fold, config) inner
evaluation, one (fold) outer prediction -- is appended to a JSONL checkpoint and
skipped on restart. Ctrl-C (or SIGTERM via the CLI) stops the run cleanly;
re-running the same command resumes where it left off.

Outputs (audit only; the production configuration in fit_models.rf_kwargs and
the wired tiers are NOT modified):
  data/rf_sweep_nested_folds__<suffix>.csv   per-fold inner winners + metrics
  data/rf_sweep_nested_preds__<suffix>.csv   per-pair nested predictions
  data/rf_sweep_nested_summary.csv           per-source nested vs global scores
Checkpoints: outputs/nested/nested_inner.jsonl, nested_outer.jsonl (gitignored).
"""
from __future__ import annotations

import argparse
import signal
import sys as _sys
import time
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

from pipeline import checkpoint as cp
from pipeline import fit_models as fm
from pipeline import sweep_rf as srf

HERE = Path(__file__).resolve().parents[1]
BASE_SEED = 42

# Committed global-selection references (rf_sweep_best.csv, combined-winner
# rows: LOLO rho/R^2 of each source's production sweep winner), the scores the
# nested run is compared against. As in floor_unseen.py's committed gate, the
# production configuration is frozen in this package rather than re-swept.
GLOBAL_SELECTION = {
    ("grambank", "wals"): (0.706178535800203, 0.4941154249469265),
    ("grambank",): (0.7796110457613202, 0.6073318302888342),
    ("wals",): (0.6544622645226708, 0.4257378420142818),
}

# One explicit output suffix per source group (always explicit, so the union is
# never written under a bare default name).
GROUP_SUFFIX = {"grambank,wals": "__grambank_wals", "grambank": "__grambank",
                "wals": "__wals", "metadata": "_metadata"}


# --- inner selection ---------------------------------------------------------
def _sub_src(src, fold_i):
    """src restricted to fold i's training rows: inner folds over the 23
    remaining languages; the outer test rows are never selected."""
    tr, te = src["folds"][fold_i]
    held = src["langs_ids"][fold_i]
    sub_langs = [g for g in src["langs_ids"] if g != held]
    sub_pairs = [src["pairs"][j] for j in tr]
    return {"langs_ids": sub_langs, "pairs": sub_pairs,
            "y": np.asarray(src["y"])[tr], "X2": src["X2"][tr],
            "folds": fm.lolo_folds(sub_pairs, sub_langs)}


def inner_metrics(src, fold_i, rf_params, *, n_estimators=200, seed=BASE_SEED,
                  holdout_k=5, holdout_repeats=5):
    """Inner LOLO + inner grouped both-unseen holdout for one config, computed
    entirely inside fold i's training pairs (outer test rows never read)."""
    sub = _sub_src(src, fold_i)
    splits = [srf.pair_idx(sub["pairs"], ts) for ts in srf.grouped_kfold_test_lang_sets(
        sub["langs_ids"], holdout_k, holdout_repeats, BASE_SEED)]
    lolo = srf.lolo_metrics(sub["X2"], sub["y"], sub["folds"], rf_params,
                            n_estimators, [seed])
    hold = srf.holdout_metrics(sub["X2"], sub["y"], splits, rf_params,
                               n_estimators, [seed])
    return {"inner_lolo_r2_mean": lolo["lolo_r2_mean"],
            "inner_lolo_rho_mean": lolo["lolo_rho_mean"],
            "inner_holdout_r2_mean": hold["holdout_r2_mean"],
            "inner_holdout_rho_mean": hold["holdout_rho_mean"]}


def pick_fold_winner(records):
    """Combined-rank winner over one fold's inner records: rank by inner LOLO
    R^2 + rank by inner holdout R^2 (production criterion), lowest sum wins;
    ties break on inner LOLO rho, then tag (determinism)."""
    d = pd.DataFrame(records)
    d["rank_lolo"] = d["inner_lolo_r2_mean"].rank(ascending=False, na_option="bottom")
    d["rank_hold"] = d["inner_holdout_r2_mean"].rank(ascending=False, na_option="bottom")
    d["rank_sum"] = d["rank_lolo"] + d["rank_hold"]
    d = d.sort_values(["rank_sum", "inner_lolo_rho_mean", "tag"],
                      ascending=[True, False, True])
    return d.iloc[0].to_dict()


# --- driver -------------------------------------------------------------------
def _outer_predict(src, fold_i, params, n_estimators=200, seed=BASE_SEED):
    tr, te = src["folds"][fold_i]
    m = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, **params)
    m.fit(src["X2"][tr], np.asarray(src["y"])[tr])
    return te, m.predict(src["X2"][te])


def _cfg_sig(configs):
    """Stable signature of the evaluated config set: an outer record is only
    resumable when the run evaluates the SAME set (a --configs-limit smoke run
    must not leave its partial-set winner in place for a full run)."""
    import hashlib
    import json as _json
    blob = _json.dumps(sorted((t, p) for t, p in configs), sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()


def latest_outer_records(records):
    """Deduplicate outer records by fold: the LAST checkpoint line wins (a fold
    re-evaluated under a larger config set supersedes the earlier record)."""
    by_fold = {}
    for r in records:
        by_fold[r["fold"]] = r
    return [by_fold[k] for k in sorted(by_fold)]


def run_nested(src, source_name, *, configs, ckpt_dir, n_jobs,
               fold_limit=None, n_estimators=200, seed=BASE_SEED,
               holdout_k=5, holdout_repeats=5):
    """Nested selection over `configs` with per-unit JSONL checkpoints.

    Returns stats {computed_inner, skipped_inner, computed_outer, skipped_outer}.
    Interrupt-safe: on KeyboardInterrupt the already-checkpointed units are kept
    and re-running resumes."""
    ckpt_dir = Path(ckpt_dir)
    inner_path, outer_path = ckpt_dir / "nested_inner.jsonl", ckpt_dir / "nested_outer.jsonl"
    inner = cp.load_jsonl(inner_path)
    outer = cp.load_jsonl(outer_path)
    sig = _cfg_sig(configs)
    stats = {"computed_inner": 0, "skipped_inner": 0,
             "computed_outer": 0, "skipped_outer": 0}
    n_folds = len(src["folds"]) if fold_limit is None else min(fold_limit, len(src["folds"]))

    for i in range(n_folds):
        fold_inner = [r for r in inner if r["source"] == source_name and r["fold"] == i]
        done_tags = {r["tag"] for r in fold_inner}
        pending = [(tag, params) for tag, params in configs if tag not in done_tags]
        stats["skipped_inner"] += len(fold_inner)
        if pending:
            print(f"[{source_name}] fold {i + 1}/{n_folds}: {len(pending)} configs to "
                  f"evaluate ({len(fold_inner)} resumed)", flush=True)
            t0 = time.time()
            try:
                stream = Parallel(n_jobs=n_jobs, backend="loky", return_as="generator")(
                    delayed(inner_metrics)(src, i, params, n_estimators=n_estimators,
                                           seed=seed, holdout_k=holdout_k,
                                           holdout_repeats=holdout_repeats)
                    for tag, params in pending)
                for (tag, params), metrics in zip(pending, stream):
                    rec = {"source": source_name, "fold": i, "tag": tag,
                           "params": params, **metrics}
                    cp.append_jsonl(inner_path, rec)
                    inner.append(rec)
                    stats["computed_inner"] += 1
            except KeyboardInterrupt:
                print(f"\n[interrupt] {stats['computed_inner']} inner units kept; "
                      f"re-run this same command to resume.", flush=True)
                raise
            print(f"[{source_name}] fold {i + 1}/{n_folds} inner done in "
                  f"{time.time() - t0:.0f}s", flush=True)

        done_outer = [r for r in outer
                      if r["source"] == source_name and r["fold"] == i
                      and r.get("cfg_sig") == sig]
        if done_outer:
            stats["skipped_outer"] += 1
            continue
        winner = pick_fold_winner(
            [r for r in inner if r["source"] == source_name and r["fold"] == i])
        te, pred = _outer_predict(src, i, winner["params"], n_estimators, seed)
        cp.append_jsonl(outer_path, {"source": source_name, "fold": i,
                                     "cfg_sig": sig,
                                     "tag": winner["tag"], "params": winner["params"],
                                     "te": [int(j) for j in te],
                                     "pred": [float(p) for p in pred]})
        stats["computed_outer"] += 1
    return stats


# --- aggregation ---------------------------------------------------------------
def load_records(ckpt_dir, kind):
    return cp.load_jsonl(Path(ckpt_dir) / f"nested_{kind}.jsonl")


def assemble_predictions(src, outer_records):
    """Fold-order last-write-wins assembly (same convention as fm.cv_predict:
    each pair is predicted in both its languages' folds, the later wins).
    Input should be `latest_outer_records` output so superseded fold records
    are dropped first."""
    y = np.asarray(src["y"], dtype=float)
    pred = np.full(len(y), np.nan)
    for r in sorted(outer_records, key=lambda r: r["fold"]):
        pred[r["te"]] = r["pred"]
    return y, pred


def summarize(src, source_name, ckpt_dir):
    y, pred = assemble_predictions(
        src, latest_outer_records(
            [r for r in load_records(ckpt_dir, "outer") if r["source"] == source_name]))
    if not np.isfinite(pred).any():
        return None
    return {"nested_rho": fm.pearson(y, pred), "nested_r2": fm.r2(y, pred)}


# --- CLI -------------------------------------------------------------------------
def _all_configs():
    return ([("default", srf.REFERENCE_CONFIGS["default"]),
             ("current_tuned", srf.REFERENCE_CONFIGS["current_tuned"])]
            + [("grid_%d" % i, p) for i, p in enumerate(srf.screen_configs())])


def _write_outputs(src, source_name, suffix, ckpt_dir, out_dir):
    inner = [r for r in load_records(ckpt_dir, "inner") if r["source"] == source_name]
    outer = latest_outer_records(
        [r for r in load_records(ckpt_dir, "outer") if r["source"] == source_name])
    if not outer:
        return None
    folds_df = pd.DataFrame([
        {"fold": r["fold"], "winner_tag": r["tag"], **r["params"],
         "te_pairs": len(r["te"])} | next(
            ({"inner_lolo_r2_mean": q["inner_lolo_r2_mean"],
              "inner_holdout_r2_mean": q["inner_holdout_r2_mean"]}
             for q in inner if q["fold"] == r["fold"] and q["tag"] == r["tag"]), {})
        for r in sorted(outer, key=lambda r: r["fold"])])
    folds_df.to_csv(out_dir / f"rf_sweep_nested_folds{suffix}.csv", index=False)
    y, pred = assemble_predictions(src, outer)
    pd.DataFrame({"source_lang": [p[0] for p in src["pairs"]],
                  "target_lang": [p[1] for p in src["pairs"]],
                  "y": y, "nested_pred": pred}).to_csv(
        out_dir / f"rf_sweep_nested_preds{suffix}.csv", index=False)
    summ = summarize(src, source_name, ckpt_dir)
    group = tuple(s.strip() for s in source_name.split(","))
    if group in GLOBAL_SELECTION:
        summ["global_rho"], summ["global_r2"] = GLOBAL_SELECTION[group]
    summ["source"] = source_name
    summ["n_folds"] = len(outer)
    summ["unique_winner_tags"] = folds_df["winner_tag"].nunique()
    return summ


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-groups", default="grambank,wals;grambank;wals;metadata",
                    help="semicolon-separated source groups; sources inside a group "
                         "are comma-separated as in sweep_rf.py; 'metadata' runs the "
                         "metadata control")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="parallel workers for the inner evaluations (-1: all cores)")
    ap.add_argument("--ckpt-dir", default=str(HERE / "outputs" / "nested"))
    ap.add_argument("--fold-limit", type=int, default=None,
                    help="cap outer folds (tests / smoke runs)")
    ap.add_argument("--configs-limit", type=int, default=None,
                    help="cap grid configs after the two references (smoke runs)")
    ap.add_argument("--n-estimators", type=int, default=200)
    args = ap.parse_args()

    def _stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _stop)

    configs = _all_configs()
    if args.configs_limit is not None:
        configs = configs[:2 + args.configs_limit]
    out_dir = HERE / "data"
    summaries = []
    for group in args.source_groups.split(";"):
        group = group.strip()
        suffix = GROUP_SUFFIX[group]
        if group == "metadata":
            from experiments.robustness_controls import _metadata_design
            langs_ids, pairs, y, folds, Xm, _ = _metadata_design()
            src = {"langs_ids": langs_ids, "pairs": pairs, "y": y,
                   "X2": Xm, "folds": folds}
            name = group
        else:
            sources = tuple(group.split(","))
            src = srf.prep_source(sources)
            name = group
        try:
            stats = run_nested(src, name, configs=configs, ckpt_dir=args.ckpt_dir,
                               n_jobs=args.n_jobs, fold_limit=args.fold_limit,
                               n_estimators=args.n_estimators)
        except KeyboardInterrupt:
            print("\n[stopped] partial progress checkpointed; re-run to resume.")
            raise SystemExit(130)
        print(f"[{name}] inner: {stats['computed_inner']} computed, "
              f"{stats['skipped_inner']} resumed | outer: "
              f"{stats['computed_outer']} computed, {stats['skipped_outer']} resumed")
        s = _write_outputs(src, name, suffix, args.ckpt_dir, out_dir)
        if s:
            summaries.append(s)
            print(f"[{name}] nested rho={s['nested_rho']:.4f} r2={s['nested_r2']:.4f}"
                  + (f" | global-selection rho={s.get('global_rho', float('nan')):.4f}"
                     f" r2={s.get('global_r2', float('nan')):.4f}"
                     f" | winners={s['unique_winner_tags']} unique"
                     if "global_rho" in s else ""))
    if summaries:
        pd.DataFrame(summaries).to_csv(out_dir / "rf_sweep_nested_summary.csv", index=False)
        print("\nsummary written to data/rf_sweep_nested_summary.csv")


if __name__ == "__main__":
    main()
