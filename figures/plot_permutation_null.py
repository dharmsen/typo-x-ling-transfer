# plot_permutation_null.py
"""Paper figure: fold-respecting permutation null vs observed LOLO scores.

Reproduces figures/permutation_null__lolo.pdf (appendix) for the tuned RF on
Grambank+WALS: two panels (left Pearson rho, right true R^2), each showing the
null distribution as bars, its mean as a dashed line, and the observed score
as a solid line.

Uses the exact protocol of infer_stats.permutation_null / run_stats (per fold,
permute the TRAINING-set y only, refit, predict held-out; rng stream:
default_rng(seed), one permutation per fold per replicate), so with the
default B=200 and seed 42 the draws and summaries match the committed
data/stats__lolo.csv run. The draws are recomputed here rather than
calling permutation_null because that helper returns summaries only.

NOTE: each null replicate refits the RF under 24 LOLO folds, so B=200 takes
hours (same cost as `experiments/run_experiments.py --exp stats`); use --B 8 for a
quick smoke test of the figure mechanics.

Run:  uv run python figures/plot_permutation_null.py [--B 200]
Writes: outputs/figures/paper/permutation_null__lolo.{pdf,png}
"""
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from experiments import run_experiments as el
from pipeline import fit_models as fm

OUT = Path("outputs/figures/paper")


def main(B=200, seed=42):
    s = el.load_setup(sources=("grambank", "wals"))
    wide = s["wide"].reindex(index=s["langs_ids"])
    M, foc, _, feats, idx = el._grambank._one_hot_layout(wide, list(wide.columns))
    X = fm.stage2_design(s["pairs"], {g: k for k, g in enumerate(idx)}, M, foc, len(feats))
    y, folds = s["y"], s["folds"]
    rff = lambda: RandomForestRegressor(n_estimators=200, random_state=42,
                                        **fm.rf_kwargs("full_tuned", ("grambank", "wals")))

    pred = fm.cv_predict(X, y, folds, rff)
    obs_r, obs_t = fm.pearson(y, pred), fm.r2(y, pred)

    print("computing permutation null (B=%d, seed=%d; each replicate is a full LOLO refit)..."
          % (B, seed))
    rng = np.random.default_rng(seed)
    draws_r, draws_t = [], []
    for _ in range(B):
        yp = y.copy()
        for tr, _ in folds:
            yp[tr] = y[tr][rng.permutation(len(tr))]
        p = fm.cv_predict(X, yp, folds, rff)
        draws_r.append(fm.pearson(y, p))
        draws_t.append(fm.r2(y, p))
    draws_r, draws_t = np.asarray(draws_r), np.asarray(draws_t)

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), constrained_layout=True)
    for ax, draws, obs, lab in ((axes[0], draws_r, obs_r, r"Pearson $\rho$"),
                                (axes[1], draws_t, obs_t, r"$R^2$")):
        ax.hist(draws, bins=25, color="0.7", edgecolor="white")
        ax.axvline(np.mean(draws), ls="--", color="0.2", label="null mean")
        ax.axvline(obs, ls="-", color="C3", lw=2, label="observed")
        ax.set_xlabel(lab)
        ax.set_ylabel("null replicates")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Fold-respecting permutation null vs observed LOLO (tuned RF, GB+WALS, "
                 "B=%d, seed %d)" % (B, seed), fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"permutation_null__lolo.{ext}", dpi=300)
    ge = int(np.sum(draws_r >= obs_r))
    print("wrote %s/permutation_null__lolo.{pdf,png}" % OUT)
    print("null mean rho %.4f | %d/%d nulls >= observed rho -> p=%.4f"
          % (np.mean(draws_r), ge, B, (ge + 1) / (B + 1)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--B", type=int, default=200, help="null replicates (default 200, slow)")
    a = ap.parse_args()
    main(B=a.B)
