# floor_unseen.py
"""Pure-unseen-language protocol across the coverage-floor sweep.

Regenerates ``data/floor_unseen.csv``: the tuned RF in its production
configuration (max_features=0.5, scikit-learn defaults otherwise, 200 trees)
scored under sweep_rf.py's grouped-holdout protocol -- repeated grouped 5-fold
over languages with 5 repeats, split seed 42, test pairs whose BOTH languages
are held out, pooled predictions per seed over seeds 42-44 -- at each floor of
the coverage-floor sweep (README, "Coverage-floor sweep"). This is the final
column of the paper's coverage-floor table (tab:floor).

The floor 0.70 row is a correctness gate: it must reproduce the committed
sweep result for the identical configuration (data/rf_sweep_best.csv,
grambank,wals holdout winner grid_49 == the production config) before any
floored row is trusted.

Usage (from this directory):
    uv run python experiments/floor_unseen.py               # all five floors, writes the CSV
    uv run python experiments/floor_unseen.py --floors 0.5  # subset, prints only (no write)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import _grambank
from pipeline import fit_models as fm
from pipeline import sweep_rf as srf

HERE = Path(__file__).resolve().parents[1]
OUT_PATH = HERE / "data" / "floor_unseen.csv"
ALL_FLOORS = (0.70, 0.60, 0.50, 0.40, 0.00)

# The production full_tuned config (fm.rf_kwargs("full_tuned")); identical to
# the sweep's grid_49 row, the committed holdout winner for grambank,wals.
PROD_PARAMS = {"max_features": 0.5, "bootstrap": True}
N_ESTIMATORS = 200
HOLDOUT_SEEDS = list(range(42, 45))
HOLDOUT_K = 5
HOLDOUT_REPEATS = 5

# Committed gate (data/rf_sweep_best.csv, sources="grambank,wals",
# winner="holdout", tag="grid_49"): the same config's floor-0.70 holdout result.
GATE_FLOOR = 0.70
GATE_R2 = 0.1356365327596696
GATE_RHO = 0.40496688990125573


def prep_floor(floor):
    """The config-independent Stage-2 design at a given coverage floor.

    sweep_rf.prep_source with the floor parameter threaded through; the design
    construction is bit-identical (fm._load_common + _one_hot_layout +
    fm.stage2_design)."""
    bts, langs_ids, langs, wide, scripts = fm._load_common(floor, ("grambank", "wals"), "gb")
    sub = wide.reindex(index=langs_ids)
    M, foc, has, feats, idx = _grambank._one_hot_layout(sub, list(wide.columns))
    pos = {g: k for k, g in enumerate(idx)}
    pairs = fm.directed_pairs(langs_ids)
    bts_idx = bts.set_index(["glottocode_s", "glottocode_t"])
    y = np.array([float(bts_idx.loc[(s, t), "bts"]) for s, t in pairs])
    X2 = fm.stage2_design(pairs, pos, M, foc, len(feats))
    return langs_ids, pairs, y, X2


def run(floors, write):
    rows = []
    for floor in floors:
        langs_ids, pairs, y, X2 = prep_floor(floor)
        splits = [srf.pair_idx(pairs, ts) for ts in srf.grouped_kfold_test_lang_sets(
            langs_ids, HOLDOUT_K, HOLDOUT_REPEATS, srf.BASE_SEED)]
        m = srf.holdout_metrics(X2, y, splits, PROD_PARAMS, N_ESTIMATORS, HOLDOUT_SEEDS)
        row = dict(floor=floor, n=len(langs_ids), pairs=len(pairs), **m)
        rows.append(row)
        print(f"floor {floor:.2f} n={row['n']} pairs={row['pairs']} "
              f"holdout R2={m['holdout_r2_mean']:+.3f}±{m['holdout_r2_sd']:.3f} "
              f"rho={m['holdout_rho_mean']:+.3f}±{m['holdout_rho_sd']:.3f}", flush=True)
        if floor == GATE_FLOOR:
            ok = (abs(m["holdout_r2_mean"] - GATE_R2) < 1e-9
                  and abs(m["holdout_rho_mean"] - GATE_RHO) < 1e-9)
            print(f"  [gate {'PASS' if ok else 'FAIL'}] floor {GATE_FLOOR:.2f} "
                  f"R2={m['holdout_r2_mean']:.10f} (expected {GATE_R2:.10f}) "
                  f"rho={m['holdout_rho_mean']:.10f} (expected {GATE_RHO:.10f})",
                  flush=True)
            if not ok:
                raise SystemExit("Correctness gate FAILED -- the protocol does not "
                                 "reproduce the committed sweep result; fix before "
                                 "trusting any floored row.")
    if not write:
        print("subset run: CSV not written")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unknown"
    meta = {
        "floors": list(floors),
        "rf_params": {"n_estimators": N_ESTIMATORS, **PROD_PARAMS},
        "holdout": {"k": HOLDOUT_K, "repeats": HOLDOUT_REPEATS,
                    "split_seed": srf.BASE_SEED, "seeds": HOLDOUT_SEEDS},
        "gate": {"floor": GATE_FLOOR, "r2": GATE_R2, "rho": GATE_RHO},
        "git_commit": commit,
    }
    (HERE / "data" / "floor_unseen_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    print(f"saved {OUT_PATH.name} + floor_unseen_meta.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--floors", type=float, nargs="+", default=None,
                    help="subset of floors to score (default: all five). "
                         "A subset only prints; the CSV is written on the full run.")
    a = ap.parse_args()
    floors = tuple(a.floors) if a.floors else ALL_FLOORS
    run(floors, write=floors == ALL_FLOORS)


if __name__ == "__main__":
    main()
