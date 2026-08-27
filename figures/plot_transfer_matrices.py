# plot_transfer_matrices.py
"""Render the ATLAS Figure C.2 transfer matrix in the paper's style, using the
typology-proxy models (similarity / glm / RandomForest / RF-top-25) instead of
the measured BTS, plus per-model difference maps against the digitized ground
truth.

Feature sources are selectable via --sources (default grambank,wals); figures
are written to source-suffixed filenames so the Grambank-only baseline figures
coexist with the Grambank+WALS ones.

Every panel is drawn on the full 38-language ATLAS footprint; the 14 languages
without Grambank data are masked (grey) in the model and difference panels.

Run:  uv run python figures/plot_transfer_matrices.py [--sources grambank,wals]
Writes: outputs/figures/{transfer_matrices[__sources].png, ground_truth.png,
       similarity|glm|rf|rf25[__sources].png, diff_*[__sources].png}
"""
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as patheffects
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from pipeline import _grambank
from pipeline import data_sources
from pipeline import fit_models as fm

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
OUT = HERE / "outputs" / "figures"

# --- language order, families, and the modelable subset --------------------

# The axis order is read straight from the ground-truth CSV: the digitization
# walked the matrix in (source-outer, target-inner) order, so first-appearance
# in the `source` column IS the ATLAS axis order.


def _load_lang_order():
    seen, order = set(), []
    with open(DATA / "blt_ground_truth.csv") as f:
        for r in csv.DictReader(f):
            if r["source"] not in seen:
                seen.add(r["source"])
                order.append(r["source"])
    return order


LANG_ORDER = _load_lang_order()


def _load_atlas_map():
    """code -> dict(glottocode, coverage) from atlas_languages.csv."""
    out = {}
    with open(DATA / "atlas_languages.csv") as f:
        for r in csv.DictReader(f):
            out[r["atlas_name"]] = {
                "glottocode": (r["glottocode"] or "").strip() or None,
                "coverage": float(r["coverage"]) if r["coverage"].strip() else 0.0,
            }
    return out


_ATLAS = _load_atlas_map()
FLOOR = 0.70

# The 24 languages with a Grambank glottocode at/above the coverage floor.
MODELABLE = frozenset(
    c for c, m in _ATLAS.items()
    if m["glottocode"] and m["coverage"] >= FLOOR
)
_GC2CODE = {m["glottocode"]: c for c, m in _ATLAS.items() if m["glottocode"]}

# Family per language, grouped at Figure C.2's legend granularity (Indo-European
# split into its sub-families; otherwise the Glottolog top-level family). The 28
# languages with Grambank data follow Glottolog; the 10 Grambank-absent ones are
# assigned from their standard classification.
FAMILY = {
    "en": "Germanic", "ru": "Slavic", "es": "Romance", "fr": "Romance",
    "de": "Germanic", "pt": "Romance", "zh": "Sinitic", "vi": "Austro-Asiatic",
    "ja": "Japonic", "sw": "Bantu", "ko": "Koreanic", "cy": "Celtic",
    "kk": "Turkic", "tr": "Turkic", "af": "Germanic", "no": "Germanic",
    "sv": "Germanic", "el": "Hellenic", "bn": "Indo-Iranian", "ckb": "Indo-Iranian",
    "fa": "Indo-Iranian", "gu": "Indo-Iranian", "mr": "Indo-Iranian",
    "ps": "Indo-Iranian", "ur": "Indo-Iranian", "fil": "Austronesian",
    "id": "Austronesian", "ms": "Austronesian", "ca": "Romance", "it": "Romance",
    "ro": "Romance", "ar": "Semitic", "he": "Semitic", "hr": "Slavic",
    "uk": "Slavic", "te": "Dravidian", "th": "Tai-Kadai", "yo": "Volta-Niger",
}

N = len(LANG_ORDER)
_POS = {c: i for i, c in enumerate(LANG_ORDER)}


# --- matrix construction ---------------------------------------------------

def gt_matrix(lang_order=LANG_ORDER):
    """38x38 BTS ground truth. Cell [target, source] = BTS(source->target).
    Diagonal is NaN (self-transfer is undefined; the CSV has no self-pairs)."""
    pos = {c: i for i, c in enumerate(lang_order)}
    m = np.full((len(lang_order), len(lang_order)), np.nan)
    with open(DATA / "blt_ground_truth.csv") as f:
        for r in csv.DictReader(f):
            m[pos[r["target"]], pos[r["source"]]] = float(r["bts"])
    return m


def to_square(preds, lang_order=LANG_ORDER, modelable=MODELABLE):
    """Place {(source, target): value} predictions into an NxN matrix keyed by
    code. Cell [target, source] = pred. Unmodelable rows/cols and the diagonal
    stay NaN."""
    pos = {c: i for i, c in enumerate(lang_order)}
    modelable = set(modelable)
    m = np.full((len(lang_order), len(lang_order)), np.nan)
    for (s, t), v in preds.items():
        if s in modelable and t in modelable and s != t:
            m[pos[t], pos[s]] = v
    return m


def fit_predictions(model, floor=FLOOR, sources=("grambank", "wals")):
    """In-sample fit of one model on the 24 modelable languages; returns
    {(source_code, target_code): predicted BTS} for every directed pair.

    Models mirror fit_models.run but fit once on all pairs (the reconstruction
    upper bound, identical to agreement.py), not under LOLO CV. `sources`
    selects the typology feature sources (passed to _load_common and, for
    rf25, the permutation-importance ranking).
    """
    bts, langs_ids, langs, wide, scripts = fm._load_common(floor, sources)
    sim = _grambank.simple_similarity(_grambank.subset_wide(wide, langs_ids))
    gene = _grambank.genealogical_distance(langs, langs_ids)
    geo = _grambank.geographic_distance(langs, langs_ids)
    pairs = fm.directed_pairs(langs_ids)
    bi = bts.set_index(["glottocode_s", "glottocode_t"])
    y = np.array([float(bi.loc[(s, t), "bts"]) for s, t in pairs])

    if model == "similarity":
        X = fm.stage1_design(pairs, sim, gene, geo, scripts)[:, [0]]
        pred = Ridge(alpha=1.0).fit(X, y).predict(X)
    elif model == "glm":
        X = fm.stage1_design(pairs, sim, gene, geo, scripts)
        pred = Ridge(alpha=1.0).fit(X, y).predict(X)
    elif model in ("rf", "rf25"):
        sub = wide.reindex(index=list(langs_ids))
        feats = fm.top_feature_ids(25, sources=sources) if model == "rf25" else list(wide.columns)
        M, foc, _, feats2, idx = _grambank._one_hot_layout(sub, feats)
        X2 = fm.stage2_design(pairs, {g: k for k, g in enumerate(idx)}, M, foc, len(feats2))
        pred = RandomForestRegressor(n_estimators=200, random_state=42).fit(X2, y).predict(X2)
    else:
        raise ValueError("unknown model: %r" % model)

    return {(_GC2CODE[s], _GC2CODE[t]): float(v) for (s, t), v in zip(pairs, pred)}


def model_matrix(model, floor=FLOOR, lang_order=LANG_ORDER, modelable=MODELABLE,
                 sources=("grambank", "wals")):
    """NxN in-sample prediction matrix for one model (masked outside the 24)."""
    return to_square(fit_predictions(model, floor, sources=sources), lang_order, modelable)


def diff_matrix(gt, model):
    """gt - model where both finite, else NaN."""
    out = np.full(gt.shape, np.nan)
    both = np.isfinite(gt) & np.isfinite(model)
    out[both] = gt[both] - model[both]
    return out


def top5_mask(mat, k=5):
    """Boolean NxN: per row (target), the k highest finite entries."""
    out = np.zeros(mat.shape, dtype=bool)
    for r in range(mat.shape[0]):
        row = mat[r]
        finite = np.where(np.isfinite(row))[0]
        if finite.size == 0:
            continue
        top = finite[np.argsort(row[finite])[::-1][:k]]
        out[r, top] = True
    return out


# --- plotting (Figure C.2 style) ------------------------------------------

_GROUPS = [g for g, _ in sorted(
    {v: None for v in FAMILY.values()}.items(), key=lambda x: x[0])]
# stable family -> color map
_PALETTE = plt.cm.tab20(np.linspace(0, 1, max(len(_GROUPS), 20)))
FAMILY_COLOR = {g: matplotlib.colors.to_hex(_PALETTE[i]) for i, g in enumerate(_GROUPS)}

_MODELS = [
    ("similarity", "similarity (Ridge·sim)"),
    ("glm", "glm (Ridge·sim+script+gene+geo)"),
    ("rf", "RandomForest (all features)"),
    ("rf25", "RandomForest (top-25 features)"),
]


def _style_axes(ax, mark_top5_mat=None, tick_fs=6):
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(LANG_ORDER, fontsize=tick_fs)
    ax.set_yticklabels(LANG_ORDER, fontsize=tick_fs)
    for tick, code in zip(ax.get_xticklabels(), LANG_ORDER):
        tick.set_color(FAMILY_COLOR[FAMILY[code]])
    for tick, code in zip(ax.get_yticklabels(), LANG_ORDER):
        tick.set_color(FAMILY_COLOR[FAMILY[code]])
    # thin grid
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", color="0.9", linewidth=0.3)
    ax.tick_params(which="minor", length=0)
    # top-5 source outlines per target row
    if mark_top5_mat is not None:
        rows, cols = np.where(mark_top5_mat)
        for r, c in zip(rows, cols):
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor="black", linewidth=0.8, zorder=5))


def _text_color(v, vmin, vmax, cmap, threshold=0.5):
    """Pick black or white text for legibility against the cell's background
    colour: white on dark (saturated) cells, black on light cells."""
    frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    frac = min(max(frac, 0.0), 1.0)
    r, g, b, _ = cmap(frac)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < threshold else "black"


def plot_heatmap(ax, mat, *, title, vmin, vmax, mark_top5=True, cmap="RdBu_r",
                 decimals=1, fontsize=4, tick_fs=6, title_fs=10):
    cmap = matplotlib.colormaps.get_cmap(cmap).copy()
    cmap.set_bad("#d9d9d9")  # masked cells (unmodelled + diagonal)
    masked = np.ma.masked_invalid(mat)
    ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal",
              interpolation="nearest")
    # numeric value in every cell; auto-contrast fill + opposite-colour stroke
    # guarantees legibility regardless of background.
    rows, cols = np.where(np.isfinite(mat))
    for r, c in zip(rows, cols):
        v = mat[r, c]
        fg = _text_color(v, vmin, vmax, cmap)
        bg = "black" if fg == "white" else "white"
        ax.text(c, r, f"{v:.{decimals}f}", ha="center", va="center",
                fontsize=fontsize, color=fg, zorder=6,
                path_effects=[patheffects.withStroke(
                    linewidth=fontsize * 0.2, foreground=bg)])
    _style_axes(ax, top5_mask(mat) if mark_top5 else None, tick_fs=tick_fs)
    ax.set_title(title, fontsize=title_fs)


def _family_legend(fig):
    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                          markerfacecolor=FAMILY_COLOR[g], markeredgecolor="none")
               for g in _GROUPS]
    fig.legend(handles, _GROUPS, loc="upper center", ncol=len(_GROUPS) // 2 + 1,
               fontsize=12, frameon=False, title="language family",
               title_fontsize=14, bbox_to_anchor=(0.5, 1.0))


def _source_label(sources):
    """Human label for a source tuple, e.g. 'Grambank' or 'Grambank+WALS'."""
    names = {"grambank": "Grambank", "wals": "WALS"}
    return "+".join(names.get(s, s) for s in sources)


def make_figure(out_dir=OUT, sources=("grambank", "wals")):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = data_sources.sources_suffix(sources)
    src_label = _source_label(sources)

    gt = gt_matrix()
    models = {key: model_matrix(key, sources=sources) for key, _ in _MODELS}
    diffs = {key: diff_matrix(gt, models[key]) for key, _ in _MODELS}

    gt_vmin, gt_vmax = float(np.nanmin(gt)), float(np.nanmax(gt))
    d_vmax = float(np.nanmax(np.abs(np.stack(list(diffs.values())))))
    d_vmax = max(d_vmax, 1e-6)
    mae = {key: float(np.nanmean(np.abs(diffs[key]))) for key, _ in _MODELS}

    # --- combined figure: row 1 = GT + 4 models, row 2 = 4 difference maps ---
    # very large canvas so 38x38 cell values and axis labels never overlap
    fig, axes = plt.subplots(2, 5, figsize=(44, 18), constrained_layout=True)
    plot_heatmap(axes[0, 0], gt, title="ground truth (digitized Fig. C.2)",
                 vmin=gt_vmin, vmax=gt_vmax, decimals=1, fontsize=5,
                 tick_fs=9, title_fs=14)
    for j, (key, label) in enumerate(_MODELS):
        plot_heatmap(axes[0, j + 1], models[key], title=label,
                     vmin=gt_vmin, vmax=gt_vmax, decimals=1, fontsize=5,
                     tick_fs=9, title_fs=14)
    for j, (key, label) in enumerate(_MODELS):
        plot_heatmap(axes[1, j], diffs[key],
                     title=f"ground truth − {key}  (MAE {mae[key]:.3f})",
                     vmin=-d_vmax, vmax=d_vmax, decimals=2, fontsize=5,
                     tick_fs=9, title_fs=14)
    axes[1, 4].axis("off")

    # shared colorbars
    fig.colorbar(plt.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(gt_vmin, gt_vmax), cmap="RdBu_r"),
        ax=axes[0, :], shrink=0.7, label="BTS  (source -> target)")
    fig.colorbar(plt.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(-d_vmax, d_vmax), cmap="RdBu_r"),
        ax=axes[1, :4], shrink=0.7, label="Δ BTS")
    _family_legend(fig)
    fig.suptitle(f"ATLAS cross-lingual transfer: ground truth vs {src_label}-proxy models",
                 fontsize=22)
    fig.savefig(out_dir / f"transfer_matrices{suffix}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- individual PNGs (diff maps use a per-model scale so each is legible) ---
    def _solo(mat, title, vmin, vmax, name, cb_label, decimals):
        fig, ax = plt.subplots(figsize=(15, 15), constrained_layout=True)
        plot_heatmap(ax, mat, title=title, vmin=vmin, vmax=vmax,
                     decimals=decimals, fontsize=7, tick_fs=10, title_fs=16)
        fig.colorbar(plt.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin, vmax), cmap="RdBu_r"),
            ax=ax, shrink=0.7, label=cb_label)
        fig.savefig(out_dir / name, dpi=200, bbox_inches="tight")
        plt.close(fig)

    _solo(gt, "ground truth (digitized Fig. C.2)", gt_vmin, gt_vmax,
          "ground_truth.png", "BTS", decimals=1)
    for key, label in _MODELS:
        _solo(models[key], label, gt_vmin, gt_vmax, f"{key}{suffix}.png", "BTS", decimals=1)
        dv = float(np.nanmax(np.abs(diffs[key])))
        _solo(diffs[key], f"ground truth − {label}  (MAE {mae[key]:.3f})",
              -dv, dv, f"diff_{key}{suffix}.png", "Δ BTS", decimals=2)

    print(f"wrote {1 + 2*len(_MODELS)} panels + combined figure to {out_dir}/")
    print(f"BTS scale: [{gt_vmin:.2f}, {gt_vmax:.2f}]   diff scale: [±{d_vmax:.2f}]")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Render ATLAS C.2-style transfer matrices for the proxy models.")
    ap.add_argument("--sources", default="grambank,wals",
                    help="comma-separated CLDF sources (default: grambank,wals; "
                         "use 'grambank' for the Grambank-only baseline figures)")
    a = ap.parse_args()
    sources = tuple(s.strip() for s in a.sources.split(","))
    make_figure(sources=sources)
