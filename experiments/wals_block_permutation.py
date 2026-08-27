"""WALS-block permutation control: is the union's drop vs Grambank noise in the
WALS data, or generic column dilution at n=24?

The WALS block of the wide feature matrix is permuted across the 24 ATLAS
languages — real WALS profiles assigned to the wrong languages — which destroys
the feature-language association while preserving each feature's marginal
levels and missingness. The Stage-2 pair design is rebuilt on the permuted
matrix and the tuned LOLO rerun with the same seven-value max_features
mini-sweep as the shared-22 control (robustness_controls.shared22), on identical
folds over the GB floor set (n=24). Comparison arms:

  grambank       195 GB columns (reference)
  grambank+wals  387 real columns (reference; the production union)
  gb+wals_perm   GB + row-permuted WALS block, K permutation seeds

If gb+wals_perm lands at the union's level, any marginally matched column block
produces the drop — dilution, not noisy data. If it lands below, real WALS
beats its own association-destroyed noise floor, so the union's modest drop is
a capacity artifact rather than noise injection.

Outputs (run from the package root):  uv run python wals_block_permutation.py
  - data/wals_block_permutation.csv   long table: one row per arm x seed x max_features
"""
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

from pipeline import _grambank
from experiments import run_experiments as el
from pipeline import fit_models as fm

SEED = 42
N_PERM = 10
# the curated sweep grid of sweep_rf.py (screening stage), as in shared22
MAX_FEATURES = [0.2, 0.3, 0.5, 0.7, 1.0, "sqrt", "log2"]


def permute_wals_block(wide, wals_cols, rng):
    """Return a copy of `wide` whose WALS rows are shuffled across languages:
    language j receives language perm[j]'s WALS profile, GB columns untouched."""
    out = wide.copy()
    perm = rng.permutation(len(wide.index))
    out.loc[:, wals_cols] = wide.loc[:, wals_cols].values[perm]
    return out


def design(setup):
    """Stage-2 [M_s, M_t, Delta] design over the setup's pairs."""
    wide = setup["wide"].reindex(index=setup["langs_ids"])
    M, foc, _, feats, idx = _grambank._one_hot_layout(wide, list(wide.columns))
    pos = {g: k for k, g in enumerate(idx)}
    X = fm.stage2_design(setup["pairs"], pos, M, foc, len(feats))
    return X, len(feats)


def main():
    s_gb = el.load_setup(sources=("grambank",))
    s_un = el.load_setup(sources=("grambank", "wals"))
    assert s_gb["langs_ids"] == s_un["langs_ids"], "GB floor set must define both arms"
    assert len(s_gb["pairs"]) == len(s_un["pairs"]) and all(
        tuple(a) == tuple(b) for a, b in zip(s_gb["pairs"], s_un["pairs"]))
    assert np.allclose(s_gb["y"], s_un["y"])
    pairs, y, folds = s_un["pairs"], s_un["y"], s_un["folds"]
    wals_cols = [c for c in s_un["wide"].columns if c not in s_gb["wide"].columns]
    assert wals_cols, "no WALS-only columns found in the union matrix"
    print(f"languages n={len(s_un['langs_ids'])}  "
          f"GB cols={s_gb['wide'].shape[1]}  WALS cols={len(wals_cols)}  "
          f"perms K={N_PERM}")

    def run(mf, X):
        pred = fm.cv_predict(X, y, folds, lambda: RandomForestRegressor(
            n_estimators=200, random_state=SEED, n_jobs=-1, max_features=mf))
        return fm.pearson(y, pred), fm.r2(y, pred)

    rows = []

    def sweep(arm, seed, X, n_feat):
        res = Parallel(n_jobs=-1, backend="loky")(delayed(run)(mf, X) for mf in MAX_FEATURES)
        for mf, (rho, r2) in zip(MAX_FEATURES, res):
            rows.append({"arm": arm, "perm_seed": seed, "n_features": n_feat,
                         "max_features": str(mf), "lolo_rho": rho, "lolo_r2": r2})
        best_i = int(np.argmax([r[1] for r in res]))   # metric of record: true R^2
        print(f"  {arm:<14} seed={seed:<3} best mf={str(MAX_FEATURES[best_i]):>5} "
              f"rho={res[best_i][0]:.4f} R2={res[best_i][1]:.4f}", flush=True)
        return res[best_i]

    X_gb, n_gb = design(s_gb)
    sweep("grambank", -1, X_gb, n_gb)
    X_un, n_un = design(s_un)
    sweep("grambank+wals", -1, X_un, n_un)

    perm_best = []
    for k in range(N_PERM):
        rng = np.random.default_rng(1000 + k)
        s_perm = dict(s_un)
        s_perm["wide"] = permute_wals_block(s_un["wide"].reindex(index=s_un["langs_ids"]),
                                            wals_cols, rng)
        X_p, n_p = design(s_perm)
        perm_best.append(sweep("gb+wals_perm", k, X_p, n_p))

    pd.DataFrame(rows).to_csv("data/wals_block_permutation.csv", index=False)

    gb = max((r for r in rows if r["arm"] == "grambank"), key=lambda r: r["lolo_r2"])
    un = max((r for r in rows if r["arm"] == "grambank+wals"), key=lambda r: r["lolo_r2"])
    pb_rho = np.array([r[0] for r in perm_best])
    pb_r2 = np.array([r[1] for r in perm_best])
    print(f"\nbest-config summary (by true R^2):")
    print(f"  grambank      rho={gb['lolo_rho']:.4f} R2={gb['lolo_r2']:.4f} "
          f"(mf={gb['max_features']})")
    print(f"  grambank+wals rho={un['lolo_rho']:.4f} R2={un['lolo_r2']:.4f} "
          f"(mf={un['max_features']})")
    print(f"  gb+wals_perm  rho={pb_rho.mean():.4f}+-{pb_rho.std(ddof=1):.4f} "
          f"[{pb_rho.min():.4f}, {pb_rho.max():.4f}]  R2={pb_r2.mean():.4f}"
          f"+-{pb_r2.std(ddof=1):.4f} (mean over {N_PERM} perms, each best-swept)")
    print("wrote data/wals_block_permutation.csv")


if __name__ == "__main__":
    main()
