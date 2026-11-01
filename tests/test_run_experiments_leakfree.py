# tests for the nested (leak-free) top-k construction of exp 3b: each fold's
# feature ranking is recomputed by inner LOLO entirely inside that fold's
# training pairs, so the held-out language touches neither the fit nor the
# ranking (the paper's corrected leak-free sweep).
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from experiments import run_experiments as el
from pipeline import _grambank
from pipeline import fit_models as fm


def _synthetic_setup(seed=0, n=8, n_feats=3):
    """Synthetic load_setup-style dict: n languages, directed pairs, one LOLO
    fold per language (integer index arrays into `pairs`)."""
    rng = np.random.default_rng(seed)
    cols = {"GB1": [frozenset({"a"})] * n}
    for f in range(2, n_feats + 1):
        cols["GB%d" % f] = rng.choice([frozenset({"x"}), frozenset({"y"})], size=n)
    wide = pd.DataFrame(cols, index=[f"g{i}" for i in range(n)])
    pairs = [np.array([wide.index[i], wide.index[j]], dtype=object)
             for i in range(n) for j in range(n) if i != j]
    folds = [([k for k, (s, t) in enumerate(pairs) if s != g and t != g],
              [k for k, (s, t) in enumerate(pairs) if s == g or t == g])
             for g in wide.index]
    return {"wide": wide, "langs_ids": list(wide.index), "pairs": pairs,
            "y": rng.normal(size=len(pairs)), "folds": folds}


def test_nested_sub_setup_excludes_heldout_language():
    """Fold i's nested ranking data is fold i's TRAINING pairs only: every inner
    language, pair, and fold excludes the held-out language entirely."""
    setup = _synthetic_setup(n=8, n_feats=3)
    held = setup["langs_ids"][0]
    sub = el._nested_sub_setup(setup, 0)
    assert sub["langs_ids"] == [g for g in setup["langs_ids"] if g != held]
    assert all(held not in p for p in sub["pairs"])
    assert len(sub["folds"]) == len(sub["langs_ids"])
    tr0 = setup["folds"][0][0]
    assert [tuple(p) for p in sub["pairs"]] == [tuple(setup["pairs"][j]) for j in tr0]
    np.testing.assert_allclose(sub["y"], setup["y"][tr0])


def test_nested_fold_ranking_passes_only_training_pairs():
    """The importance computation must receive a setup whose pairs contain no
    pair of the held-out language (the ranking is recomputed entirely inside
    each outer training set)."""
    setup = _synthetic_setup(n=6, n_feats=3)
    seen = []

    def spy(sub, feat_cols, model_factory=None, seed=42):
        seen.append(sub)
        return pd.DataFrame(0.5, index=feat_cols, columns=["f0"])

    monkey = pytest.MonkeyPatch()
    monkey.setattr(el, "_per_fold_importance", spy)
    try:
        el._nested_fold_ranking(setup, 0, list(setup["wide"].columns))
    finally:
        monkey.undo()
    assert len(seen) == 1
    held = setup["langs_ids"][0]
    assert all(held not in p for p in seen[0]["pairs"])


def test_nested_ranking_ignores_heldout_fold_targets():
    """Poisoning the held-out pairs' targets must not move fold i's ranking at
    all: those rows are never read."""
    setup = _synthetic_setup(n=6, n_feats=3)
    cols = list(setup["wide"].columns)
    r_clean = el._nested_fold_ranking(setup, 0, cols)
    poisoned = dict(setup)
    y2 = np.array(setup["y"], copy=True)
    y2[setup["folds"][0][1]] = 999.0
    poisoned["y"] = y2
    r_poison = el._nested_fold_ranking(poisoned, 0, cols)
    pd.testing.assert_series_equal(r_clean, r_poison)


def test_leakfree_topk_lolo_full_k_reduces_to_standard_lolo():
    """With k = all features, every fold selects everything, so the leak-free
    pipeline must reproduce plain cv_predict on the full design."""
    setup = _synthetic_setup()
    rankings = [el._nested_fold_ranking(setup, i, list(setup["wide"].columns))
                for i in range(len(setup["folds"]))]
    factory = lambda: Ridge(alpha=1.0)
    preds = el._leakfree_topk_lolo(setup, rankings, k=3, factory=factory)
    M, foc, _, feats, idx = _grambank._one_hot_layout(
        setup["wide"], list(setup["wide"].columns))
    X = fm.stage2_design(setup["pairs"], {g: i for i, g in enumerate(idx)}, M, foc, len(feats))
    expected = fm.cv_predict(X, setup["y"], setup["folds"], factory)
    np.testing.assert_allclose(preds, expected)
    assert np.isfinite(preds).all()


def test_leakfree_topk_k1_trains_on_own_nested_ranking():
    """k=1: fold i's model uses fold i's OWN nested top-1 feature, which can
    differ from every other fold's choice. Expectations follow the same manual
    per-fold loop and cv_predict's last-write-wins convention."""
    setup = _synthetic_setup(n=6, n_feats=2)
    rankings = [el._nested_fold_ranking(setup, i, list(setup["wide"].columns))
                for i in range(len(setup["folds"]))]
    preds = el._leakfree_topk_lolo(setup, rankings, k=1,
                                   factory=lambda: Ridge(alpha=1.0))
    wide = setup["wide"]
    expected = np.full(len(setup["y"]), np.nan)
    for i, (tr, te) in enumerate(setup["folds"]):
        feat = rankings[i].index[0]
        M, foc, _, feats, idx = _grambank._one_hot_layout(wide, [feat])
        X = fm.stage2_design(setup["pairs"], {g: j for j, g in enumerate(idx)},
                             M, foc, len(feats))
        m = Ridge(alpha=1.0).fit(X[tr], setup["y"][tr])
        expected[te] = m.predict(X[te])
    np.testing.assert_allclose(preds, expected)
    assert np.isfinite(preds).all()


def test_seed_stability_row_shape_and_determinism():
    """The seed-stability row: n_seeds tracked, all values finite correlations,
    and the computation is deterministic under a fixed factory."""
    setup = _synthetic_setup(n=6, n_feats=3)
    rankings = [el._nested_fold_ranking(setup, i, list(setup["wide"].columns))
                for i in range(len(setup["folds"]))]
    factory = lambda seed: (lambda: Ridge(alpha=1.0))   # seed-independent model
    row = el._seed_stability(setup, rankings, seeds=(42, 43, 44), k=2,
                             factory=factory)
    assert row["n_seeds"] == 3
    for key in ("full_mean", "full_std", "leakfree_mean", "leakfree_std"):
        assert np.isfinite(row[key]) and -1.0 <= row[key] <= 1.0
    # a seed-independent factory makes every replicate identical: std == 0
    assert row["full_std"] == pytest.approx(0.0, abs=1e-12)
    assert row["leakfree_std"] == pytest.approx(0.0, abs=1e-12)
    again = el._seed_stability(setup, rankings, seeds=(42, 43, 44), k=2,
                               factory=factory)
    assert row == again
