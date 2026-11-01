#!/usr/bin/env python3
# nested_paired.py
"""Paired cluster bootstrap of the nested typology-vs-metadata rho difference.

The paper's matched-tuning control reports that under fully nested selection
the typology and metadata models converge: rho 0.580 vs 0.571, a paired
difference of +0.009 at 95% CI [-0.17, 0.24]. This step regenerates that
comparison from the nested per-pair prediction tables written by
experiments/nested_tuning.py: a cluster bootstrap over source languages
(same block set resampled for both models) of LOLO Pearson(y, pred_typology)
minus Pearson(y, pred_metadata), using pipeline.infer_stats.paired_test at
B = 2000, seed 42.

Writes data/rf_sweep_nested_paired.csv.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)

import pandas as pd
from pathlib import Path

from pipeline import infer_stats

HERE = Path(__file__).resolve().parents[1]
B_DEFAULT = 2000
SEED = 42
PREDS_A = "rf_sweep_nested_preds__grambank_wals.csv"
PREDS_B = "rf_sweep_nested_preds_metadata.csv"
OUT_PATH = HERE / "data" / "rf_sweep_nested_paired.csv"


def paired_from_preds(df, pred_col_a="nested_pred", pred_col_b="nested_pred",
                      blocks_col="source_lang", B=B_DEFAULT, seed=SEED):
    """Paired cluster bootstrap of pearson(y, a) - pearson(y, b) over one
    nested-preds table holding both prediction columns. Returns one result
    row (dict) from infer_stats.paired_test."""
    res = infer_stats.paired_test(df["y"].to_numpy(dtype=float),
                                  df[pred_col_a].to_numpy(dtype=float),
                                  df[pred_col_b].to_numpy(dtype=float),
                                  df[blocks_col].to_numpy(), B=B, seed=seed)
    return {"metric": "pearson", "B": res["B"], **res}


def _load_aligned(data_dir):
    """The two nested-preds tables, asserted to cover the same pairs and y."""
    a = pd.read_csv(data_dir / PREDS_A)
    b = pd.read_csv(data_dir / PREDS_B)
    keys = ["source_lang", "target_lang", "y"]
    pd.testing.assert_frame_equal(a[keys], b[keys])
    df = a[keys].copy()
    df["nested_pred_typology"] = a["nested_pred"]
    df["nested_pred_metadata"] = b["nested_pred"]
    return df


def main():
    df = _load_aligned(HERE / "data")
    res = paired_from_preds(df, "nested_pred_typology", "nested_pred_metadata")
    out = pd.DataFrame([{**res,
                         "model_a": "grambank+wals (nested)",
                         "model_b": "metadata (nested)"}])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(out.to_string(index=False))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
