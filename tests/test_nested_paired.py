# tests for nested_paired.py: the paired cluster bootstrap of the nested
# typology-vs-metadata rho difference (the paper's Delta rho = +0.009,
# 95% CI [-0.17, 0.24]).
import numpy as np
import pandas as pd
import pytest

from experiments import nested_paired as npd
from pipeline import fit_models as fm


def _preds_df(seed=0, n_blocks=6, per_block=8, noise=0.05):
    """Synthetic nested-preds table: y is a noisy linear signal; pred_a tracks
    it, pred_b tracks it with more noise, so pearson(y, a) > pearson(y, b)."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_blocks):
        for _ in range(per_block):
            y = rng.normal()
            rows.append({"source_lang": "s%d" % b, "target_lang": "t",
                         "y": y,
                         "nested_pred_a": y + noise * rng.normal(),
                         "nested_pred_b": 4 * noise * rng.normal()})
    return pd.DataFrame(rows)


def test_paired_estimate_is_observed_rho_difference():
    df = _preds_df()
    res = npd.paired_from_preds(df, pred_col_a="nested_pred_a",
                                pred_col_b="nested_pred_b")
    want = (fm.pearson(df["y"], df["nested_pred_a"])
            - fm.pearson(df["y"], df["nested_pred_b"]))
    assert res["estimate"] == pytest.approx(want)
    assert res["B"] == npd.B_DEFAULT
    assert res["ci_lo"] < res["ci_hi"]
    assert 0.0 <= res["p_value"] <= 1.0


def test_paired_is_deterministic_under_fixed_seed():
    df = _preds_df(seed=3)
    r1 = npd.paired_from_preds(df, pred_col_a="nested_pred_a",
                               pred_col_b="nested_pred_b", B=200)
    r2 = npd.paired_from_preds(df, pred_col_a="nested_pred_a",
                               pred_col_b="nested_pred_b", B=200)
    assert r1 == r2


def test_paired_swapping_models_flips_the_sign():
    df = _preds_df(seed=5)
    r = npd.paired_from_preds(df, pred_col_a="nested_pred_a",
                              pred_col_b="nested_pred_b", B=200)
    r_swap = npd.paired_from_preds(df, pred_col_a="nested_pred_b",
                                   pred_col_b="nested_pred_a", B=200)
    assert r_swap["estimate"] == pytest.approx(-r["estimate"])
