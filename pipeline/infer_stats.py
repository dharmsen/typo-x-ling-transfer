# infer_stats.py
"""Inferential statistics over held-out (LOLO) predictions, for the paper's research questions.

Three primitives, all reusing fit_models' LOLO machinery:
  - permutation_null: fold-respecting label-shuffle null -> p(observed >= chance).
  - bootstrap_ci:     cluster (language) bootstrap CI on a metric, no refit.
  - paired_test:      cluster bootstrap of the metric DIFFERENCE between two models.

permutation_null refits the model per fold on permuted TRAINING labels (held-out
untouched -> leak-free). The CI/test operate on fixed held-out predictions with a
block (cluster) label per point (typically the source glottocode of the pair), so
they are fast and respect pair-sharing-language dependence."""
from __future__ import annotations

import numpy as np

from pipeline import fit_models as fm


def permutation_null(X, y, folds, factory, metric=None, B=1000, seed=42):
    """p(observed metric >= chance). Per fold, permute the TRAINING-set y only
    (held-out untouched), refit, predict held-out; compare held-out predictions
    to the TRUE y. Returns {observed, null_mean, p_value, B}.

    Note: with overlapping LOLO training sets, later folds' permutations win in
    the overlap, which is conservative (inflates the null / larger p-values) --
    the same pattern as fit_models.permutation_importance."""
    metric = metric or fm.pearson
    y = np.asarray(y, dtype=float)
    observed = metric(y, fm.cv_predict(X, y, folds, factory))
    if not np.isfinite(observed):
        return {"observed": float(observed), "null_mean": float("nan"),
                "p_value": float("nan"), "B": int(B)}
    rng = np.random.default_rng(seed)
    null = np.empty(int(B))
    for b in range(int(B)):
        yp = y.copy()
        for tr, _ in folds:
            yp[tr] = y[tr][rng.permutation(len(tr))]
        null[b] = metric(y, fm.cv_predict(X, yp, folds, factory))
    p = float((np.nansum(null >= observed) + 1) / (int(B) + 1))
    return {"observed": float(observed), "null_mean": float(np.nanmean(null)),
            "p_value": p, "B": int(B)}


def _cluster_resample(y_true, y_pred, blocks, rng):
    uniq = np.unique(blocks)
    samp = rng.choice(uniq, size=len(uniq), replace=True)
    idx = np.concatenate([np.where(blocks == s)[0] for s in samp])
    return y_true[idx], y_pred[idx]


def bootstrap_ci(y_true, y_pred, blocks, metric=None, B=2000, seed=42):
    """Cluster-bootstrap CI on metric(y_true, y_pred). `blocks` labels each point
    by its cluster (e.g. source glottocode). Returns {estimate, ci_lo, ci_hi, B}."""
    metric = metric or fm.pearson
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    blocks = np.asarray(blocks)
    observed = metric(y_true, y_pred)
    if not np.isfinite(observed):
        return {"estimate": float(observed), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "B": int(B)}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(B)):
        yt, yp = _cluster_resample(y_true, y_pred, blocks, rng)
        m = metric(yt, yp)
        if np.isfinite(m):
            vals.append(m)
    if not vals:
        return {"estimate": float(observed), "ci_lo": float("nan"), "ci_hi": float("nan"), "B": int(B)}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"estimate": float(observed), "ci_lo": float(lo), "ci_hi": float(hi), "B": int(B)}


def paired_test(y_true, pred_a, pred_b, blocks, metric=None, B=2000, seed=42):
    """Cluster-bootstrap test of metric(y, pred_a) - metric(y, pred_b). Each
    iteration resamples the SAME block set and applies it to y/pred_a/pred_b
    (a proper paired bootstrap). Returns {estimate, ci_lo, ci_hi, p_value, B};
    p is two-sided from the bootstrap mass crossing zero.

    Note: p is a percentile-inclusion (approximate) value -- the fraction of
    bootstrap mass crossing zero -- not an exact recentered-null test."""
    metric = metric or fm.pearson
    y_true = np.asarray(y_true, dtype=float)
    pa = np.asarray(pred_a, dtype=float)
    pb = np.asarray(pred_b, dtype=float)
    blocks = np.asarray(blocks)
    obs = metric(y_true, pa) - metric(y_true, pb)
    if not np.isfinite(obs):
        return {"estimate": float(obs), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_value": float("nan"), "B": int(B)}
    rng = np.random.default_rng(seed)
    uniq = np.unique(blocks)
    diffs = []
    for _ in range(int(B)):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(blocks == s)[0] for s in samp])
        d = metric(y_true[idx], pa[idx]) - metric(y_true[idx], pb[idx])
        if np.isfinite(d):
            diffs.append(d)
    if not diffs:
        return {"estimate": float(obs), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_value": float("nan"), "B": int(B)}
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    n = len(diffs)
    p = 2.0 * min((float(np.sum(diffs <= 0)) + 1) / (n + 1),
                  (float(np.sum(diffs >= 0)) + 1) / (n + 1))
    return {"estimate": float(obs), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_value": min(p, 1.0), "B": int(B)}
