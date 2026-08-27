# plot_paper_transfer_matrices.py
"""Paper-quality ATLAS transfer-matrix heatmaps (appendix figures).

Nine single-panel vector figures for the paper's appendix: the digitized
ATLAS-38 ground truth, the four method.tex model tiers (similarity / GLM /
full RF / tuned RF) as leave-one-language-out (LOLO) prediction matrices, and
the four ground-truth-minus-prediction difference maps.

Unlike plot_transfer_matrices.py (exploration canvases, in-sample fits), this
renders one panel per LaTeX figure at \\linewidth (~5.5 in): PDF + preview PNG,
per-cell values in auto-contrast text (black/white by background luminance +
opposite-colour stroke, as in plot_transfer_matrices.plot_heatmap), one shared
BTS color scale (the GT range) for GT and all tier panels, one shared
symmetric scale for all diff panels.

Run:  uv run python figures/plot_paper_transfer_matrices.py   (~5 min: two RF LOLO sweeps)
Writes: outputs/figures/paper/{ground_truth,similarity,glm,rf_full,rf_tuned,
       diff_similarity,diff_glm,diff_rf_full,diff_rf_tuned}.{pdf,png}
"""
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects

# LaTeX text rendering so panel text matches the paper's Computer Modern body
# font (LaTeX's default serif -- no font override or package needed). Rendered
# strings must stay TeX-safe: dashes as ---, symbols as math ($\cdot$ etc.).
# EXCEPTION: per-cell values render with usetex=False + the bundled cmr10 font
# (also Computer Modern): mpl's usetex Agg path misplaces glyphs by an amount
# that grows toward smaller x (verified: "-1.6" at column 2 sits ~1.5 glyphs
# right of center, centered at column 35), while the FreeType cmr10 path is
# exact at every column. See render_panel.
plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from pipeline import _grambank
from pipeline import fit_models as fm
from figures.plot_transfer_matrices import (
    LANG_ORDER, N, FAMILY, FAMILY_COLOR, _GC2CODE, _text_color,
    gt_matrix, to_square, diff_matrix,
)

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "outputs" / "figures" / "paper"
SOURCES = ("grambank", "wals")  # canonical paper run; no variants by design

TIERS = [
    ("similarity", "similarity proxy (Ridge$\\cdot$sim)"),
    ("glm", "GLM (Ridge$\\cdot$sim+script+gene+geo)"),
    ("rf_full", "full RandomForest"),
    ("rf_tuned", "tuned RandomForest"),
]


def lolo_predictions(tier, floor=0.70, sources=SOURCES):
    """({(source_code, target_code): prediction}, rho, mae) for one tier.

    Mirrors experiments.load_setup / rf_predict exactly (features, folds,
    seed 42, rf_kwargs('full_tuned')), so the tuned tier's rho reproduces the
    committed headline lolo_pearson = 0.705. Tier 'rf_full' is the UNTUNED RF
    (default max_features=1.0) matching fit_models' 'full' baseline.
    """
    bts, langs_ids, langs, wide, scripts = fm._load_common(floor, sources)
    sim = _grambank.simple_similarity(_grambank.subset_wide(wide, langs_ids))
    gene = _grambank.genealogical_distance(langs, langs_ids)
    geo = _grambank.geographic_distance(langs, langs_ids)
    pairs = fm.directed_pairs(langs_ids)
    bi = bts.set_index(["glottocode_s", "glottocode_t"])
    y = np.array([float(bi.loc[(s, t), "bts"]) for s, t in pairs])
    folds = fm.lolo_folds(pairs, langs_ids)

    if tier == "similarity":
        X = fm.stage1_design(pairs, sim, gene, geo, scripts)[:, [0]]
        pred = fm.cv_predict(X, y, folds, lambda: Ridge(alpha=1.0))
    elif tier == "glm":
        X = fm.stage1_design(pairs, sim, gene, geo, scripts)
        pred = fm.cv_predict(X, y, folds, lambda: Ridge(alpha=1.0))
    elif tier in ("rf_full", "rf_tuned"):
        sub = wide.reindex(index=list(langs_ids))
        M, foc, _, feats, idx = _grambank._one_hot_layout(sub, list(wide.columns))
        X = fm.stage2_design(pairs, {g: k for k, g in enumerate(idx)}, M, foc, len(feats))
        kw = {"n_estimators": 200, "random_state": 42}
        if tier == "rf_tuned":
            kw.update(fm.rf_kwargs("full_tuned", sources, "gb"))
        pred = fm.cv_predict(X, y, folds, lambda: RandomForestRegressor(**kw))
    else:
        raise ValueError("unknown tier: %r" % tier)

    rho = fm.pearson(pred, y)
    mae = float(np.mean(np.abs(np.asarray(pred) - y)))
    return {(_GC2CODE[s], _GC2CODE[t]): float(v) for (s, t), v in zip(pairs, pred)}, rho, mae


def render_panel(mat, stem, title, vmin, vmax, *, cmap="RdBu_r", cb_label,
                 cell_decimals=None, cell_fs=4.0):
    """One appendix panel: square heatmap + colorbar + family-colored ticks.

    Paper sizing: ~5.5in of plot width inside a 6.0in figure so `width=\\linewidth`
    in LaTeX renders text at true point size. With cell_decimals set, every
    finite cell carries its value in auto-contrast text (plot_transfer_matrices
    convention: fg black/white by cell luminance, stroked in the opposite
    colour), legible on any background; identity of axes is carried by the
    codes, with family color as the grouping cue (never color-alone).
    """
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 5.4), constrained_layout=True)
    cm = matplotlib.colormaps.get_cmap(cmap).copy()
    cm.set_bad("#d9d9d9")  # unmodelled rows/cols (14 langs) + diagonal
    ax.imshow(np.ma.masked_invalid(mat), cmap=cm, vmin=vmin, vmax=vmax,
              aspect="equal", interpolation="nearest")
    if cell_decimals is not None:
        rows, cols = np.where(np.isfinite(mat))
        for r, c in zip(rows, cols):
            v = mat[r, c]
            fg = _text_color(v, vmin, vmax, cm)
            bg = "black" if fg == "white" else "white"
            ax.text(c, r, f"{v:.{cell_decimals}f}", ha="center", va="center",
                    fontsize=cell_fs, color=fg, zorder=6,
                    usetex=False, fontfamily="cmr10",
                    path_effects=[patheffects.withStroke(
                        linewidth=cell_fs * 0.2, foreground=bg)])
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(LANG_ORDER, fontsize=5.5, rotation=90)
    ax.set_yticklabels(LANG_ORDER, fontsize=5.5)
    for tick, code in zip(ax.get_xticklabels(), LANG_ORDER):
        tick.set_color(FAMILY_COLOR[FAMILY[code]])
    for tick, code in zip(ax.get_yticklabels(), LANG_ORDER):
        tick.set_color(FAMILY_COLOR[FAMILY[code]])
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(
        matplotlib.colors.Normalize(vmin, vmax), cmap=cmap), ax=ax,
        shrink=0.85, label=cb_label)
    groups = sorted(set(FAMILY.values()))
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=5,
                          markerfacecolor=FAMILY_COLOR[g], markeredgecolor="none")
               for g in groups]
    # Family legend at (at most) the matrix width: pick the widest column count
    # whose drawn legend fits inside the axes, so the legend never widens the
    # tight canvas (ncol=8 measured 1.67x the axes and blew the saved figure
    # out to 7.6in). More rows, fewer columns, near-square final figure.
    leg = None
    for ncol in range(6, 1, -1):
        if leg is not None:
            leg.remove()
        leg = fig.legend(handles, groups, loc="outside lower center", ncol=ncol,
                         fontsize=6.5, frameon=False, title="language family",
                         title_fontsize=7)
        fig.canvas.draw()
        ren = fig.canvas.get_renderer()
        if leg.get_window_extent(ren).width <= ax.get_window_extent(ren).width:
            break
    print(f"  [{stem}] legend ncol={ncol} rows={-(-len(groups) // ncol)}")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    gt = gt_matrix()
    vmin, vmax = float(np.nanmin(gt)), float(np.nanmax(gt))

    preds, diffs, stats = {}, {}, {}
    for key, _ in TIERS:
        cells, rho, mae = lolo_predictions(key)
        preds[key] = to_square(cells)
        diffs[key] = diff_matrix(gt, preds[key])
        stats[key] = (rho, mae)
    d_vmax = max(float(np.nanmax(np.abs(diffs[k]))) for k, _ in TIERS)

    render_panel(gt, "ground_truth",
                 "ATLAS-38 ground truth (digitized Fig. C.2)",
                 vmin, vmax, cb_label="BTS (source $\\rightarrow$ target)",
                 cell_decimals=1, cell_fs=4.0)
    for key, label in TIERS:
        render_panel(preds[key], key, f"{label} --- LOLO predictions",
                     vmin, vmax, cb_label="BTS (source $\\rightarrow$ target)",
                     cell_decimals=1, cell_fs=4.0)
    for key, label in TIERS:
        render_panel(diffs[key], f"diff_{key}", f"ground truth $-$ {label}",
                     -d_vmax, d_vmax, cb_label="$\\Delta$ BTS = gt $-$ prediction",
                     cell_decimals=2, cell_fs=3.0)

    print(f"wrote 9 panels (.pdf + .png) to {OUT}/")
    print("--- caption block (paste into sections/appendix/transfer_matrices.tex) ---")
    print(f"BTS scale [{vmin:.2f}, {vmax:.2f}]   diff scale ±{d_vmax:.2f}")
    for key, _ in TIERS:
        rho, mae = stats[key]
        print(f"{key:10s} LOLO rho = {rho:.3f}   MAE = {mae:.3f} BTS")
    print("sanity anchor: rf_tuned rho must reproduce the committed 0.705")


if __name__ == "__main__":
    main()
