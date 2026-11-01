# tests for nested_tuning.py: the nested hyperparameter selection (selection
# recomputed inside each outer training set; outer scores never touch the
# selection data), with interrupt/resume checkpoints.
import numpy as np
import pandas as pd

from experiments import nested_tuning as ns
from pipeline import _grambank
from pipeline import fit_models as fm


def _synthetic_src(seed=0, n=8, n_feats=4):
    """src dict shaped like sweep_rf.prep_source output, on synthetic data:
    y is a noisy linear function of the design so metrics are finite."""
    rng = np.random.default_rng(seed)
    cols = {"GB1": [frozenset({"a"})] * n}
    for f in range(2, n_feats + 1):
        cols["GB%d" % f] = rng.choice([frozenset({"x"}), frozenset({"y"})], size=n)
    wide = pd.DataFrame(cols, index=[f"g{i}" for i in range(n)])
    langs = list(wide.index)
    pairs = fm.directed_pairs(langs)
    M, foc, _, feats, idx = _grambank._one_hot_layout(wide, list(wide.columns))
    pos = {g: i for i, g in enumerate(idx)}
    X2 = fm.stage2_design(pairs, pos, M, foc, len(feats))
    y = X2 @ rng.normal(size=X2.shape[1]) * 0.5 + rng.normal(size=len(pairs)) * 0.1
    folds = fm.lolo_folds(pairs, langs)
    return dict(langs_ids=langs, pairs=pairs, y=y, X2=X2, folds=folds)


TWO_CONFIGS = [("mf_half", {"max_features": 0.5, "bootstrap": True}),
               ("default", {"bootstrap": True})]


def test_inner_metrics_never_read_outer_test_rows():
    """Poisoning the outer fold's test rows (both X and y) must leave the inner
    metrics bit-identical: the inner LOLO and inner holdout run entirely on the
    outer training set."""
    src = _synthetic_src()
    params = {"max_features": 0.5, "bootstrap": True}
    clean = ns.inner_metrics(src, 0, params, holdout_k=2, holdout_repeats=1)
    tr, te = src["folds"][0]
    poisoned = dict(src)
    y2 = np.array(src["y"], copy=True)
    y2[te] = 999.0
    X2 = np.array(src["X2"], copy=True)
    X2[te, :] = 0.0
    poisoned["y"], poisoned["X2"] = y2, X2
    poiso = ns.inner_metrics(poisoned, 0, params, holdout_k=2, holdout_repeats=1)
    assert clean == poiso


def test_pick_fold_winner_uses_combined_rank_sum():
    """Winner = lowest rank(inner lolo R2) + rank(inner holdout R2); ties break
    on inner lolo rho."""
    recs = [
        {"tag": "a", "inner_lolo_r2_mean": 0.30, "inner_holdout_r2_mean": 0.10, "inner_lolo_rho_mean": 0.5},
        {"tag": "b", "inner_lolo_r2_mean": 0.20, "inner_holdout_r2_mean": 0.40, "inner_lolo_rho_mean": 0.6},
        {"tag": "c", "inner_lolo_r2_mean": 0.10, "inner_holdout_r2_mean": 0.30, "inner_lolo_rho_mean": 0.9},
    ]
    # ranks (lolo): a=1 b=2 c=3 ; (holdout): b=1 c=2 a=3 -> sums a=4 b=3 c=5
    assert ns.pick_fold_winner(recs)["tag"] == "b"
    recs[1]["inner_holdout_r2_mean"] = 0.10  # sums a=4, b=5 (rho tiebreak a), c=5
    assert ns.pick_fold_winner(recs)["tag"] == "a"


def test_run_nested_resumes_without_recompute(tmp_path):
    """Second invocation with the same checkpoint dir performs zero inner and
    zero outer work units and still returns the assembled predictions."""
    src = _synthetic_src(n=6)
    kw = dict(configs=TWO_CONFIGS, ckpt_dir=tmp_path, n_jobs=2,
              holdout_k=2, holdout_repeats=1)
    first = ns.run_nested(src, "synthetic", **kw)
    assert first["computed_inner"] == 2 * len(src["folds"])
    assert first["computed_outer"] == len(src["folds"])
    recs = [r for r in ns.load_records(tmp_path, "outer") if r["source"] == "synthetic"]
    y1, p1 = ns.assemble_predictions(src, recs)
    second = ns.run_nested(src, "synthetic", **kw)
    assert second["computed_inner"] == 0
    assert second["computed_outer"] == 0
    y2, p2 = ns.assemble_predictions(src, recs)
    np.testing.assert_allclose(p1, p2)
    assert np.isfinite(p1).all()
    assert fm.pearson(y1, p1) == fm.pearson(y2, p2)


def test_outer_recomputed_when_config_set_grows(tmp_path):
    """An outer record computed from a PARTIAL config set (e.g. a --configs-limit
    smoke run) must be recomputed, not resumed, once the full set is evaluated."""
    src = _synthetic_src(n=6)
    one = TWO_CONFIGS[:1]
    ns.run_nested(src, "synthetic", configs=one, ckpt_dir=tmp_path, n_jobs=2,
                  holdout_k=2, holdout_repeats=1)
    grown = ns.run_nested(src, "synthetic", configs=TWO_CONFIGS, ckpt_dir=tmp_path,
                          n_jobs=2, holdout_k=2, holdout_repeats=1)
    assert grown["computed_inner"] == len(src["folds"])  # only the new config
    assert grown["computed_outer"] == len(src["folds"])  # outer redone per fold
    outer = ns.latest_outer_records(
        [r for r in ns.load_records(tmp_path, "outer") if r["source"] == "synthetic"])
    for fold in range(len(src["folds"])):
        inner = [r for r in ns.load_records(tmp_path, "inner")
                 if r["source"] == "synthetic" and r["fold"] == fold]
        assert outer[fold]["params"] == ns.pick_fold_winner(inner)["params"]


def test_outer_record_params_match_inner_winner(tmp_path):
    """After a run, each fold's outer prediction must come from the config the
    INNER selection picked for that fold (recomputed from the checkpoint)."""
    src = _synthetic_src(n=6)
    ns.run_nested(src, "synthetic", configs=TWO_CONFIGS, ckpt_dir=tmp_path,
                  n_jobs=2, holdout_k=2, holdout_repeats=1)
    inner = [r for r in ns.load_records(tmp_path, "inner")
             if r["source"] == "synthetic"]
    outer = [r for r in ns.load_records(tmp_path, "outer")
             if r["source"] == "synthetic"]
    for fold in range(len(src["folds"])):
        want = ns.pick_fold_winner([r for r in inner if r["fold"] == fold])
        got = [r for r in outer if r["fold"] == fold][0]
        assert got["params"] == want["params"]
