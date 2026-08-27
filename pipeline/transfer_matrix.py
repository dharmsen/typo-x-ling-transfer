# transfer_matrix.py
"""D4: predict the BTS transfer matrix for an open universe of languages.

Resolves a language universe from typological feature coverage, fits the
validated full_tuned RandomForest on the ATLAS-24 training set, and predicts
BTS for every directed pair in the universe -- routing each cell to the RF
(extrapolation in feature space, since the Stage-2 encoding is feature-value-
based) or, optionally, to the ridge generalizer. Every cell is annotated with
the model that served it, whether it is an extrapolation beyond the ATLAS
training set, and a confidence score.

RF predictions beyond the ATLAS training set are UNVALIDATED (no BTS ground
truth exists outside ATLAS). LOLO rho ~= 0.78 is the in-distribution estimate;
weight off-distribution cells by the 'confidence' column.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from pipeline import _grambank
from pipeline import atlas_languages
from pipeline import data_sources
from pipeline import fit_models as fm

HERE = Path(__file__).resolve().parents[1]
CHUNK = 20000  # max pairs encoded+predicted per RF batch (bounds memory)


def per_source_coverage(sources):
    """{source: pd.Series(glottocode -> fraction of features present)} over the
    FULL source wide matrix (universe-scale, not ATLAS-bounded)."""
    out = {}
    for s in sources:
        w = data_sources._load_source_wide(s)
        out[s] = w.notna().sum(axis=1) / max(w.shape[1], 1)
    return out


def _mode_passes(g, coverage, mode, threshold):
    """Pure OR/AND/GB filter over per-source coverage (a source a glottocode is
    absent from counts as coverage 0.0). gb = Grambank coverage >= threshold
    (requires the 'grambank' source); union = ANY source; intersect = ALL."""
    if mode == "gb":
        gb = coverage.get("grambank")
        if gb is None:
            raise ValueError("lang-set 'gb' requires the 'grambank' source")
        return float(gb.get(g, 0.0)) >= threshold
    passes = [float(coverage[s].get(g, 0.0)) >= threshold for s in coverage]
    if mode == "union":
        return any(passes)
    if mode == "intersect":
        return all(passes)
    raise ValueError(f"unknown mode {mode!r}; expected 'gb', 'union', or 'intersect'")


def resolve_universe(coverage, all_glottocodes, mode, threshold,
                     preset=None, languages=None, exclude=None):
    """Ordered glottocode list of the matrix axes (pure function).

    The candidate pool starts from preset ∪ languages (if either given),
    otherwise from all_glottocodes; it is intersected with all_glottocodes (a
    language the sources do not attest cannot be predicted). `mode` + `threshold`
    filter the pool; `exclude` is applied last. Returns a sorted list."""
    pool = set(all_glottocodes)
    if preset is not None or languages is not None:
        candidates = (preset or set()) | (languages or set())
    else:
        candidates = set(all_glottocodes)
    candidates &= pool
    out = [g for g in candidates if _mode_passes(g, coverage, mode, threshold)]
    if exclude:
        out = [g for g in out if g not in exclude]
    return sorted(out)


def feature_code_columns(atlas_sub, feat_cols):
    """Reconstruct _one_hot_layout's one-hot column mapping from the ATLAS
    training subset, so non-ATLAS languages can be encoded into the SAME column
    space the RF was validated on.

    Returns (col_of, feat_of_code, n_codes):
      col_of      : {(feat_index, code) -> global_column}
      feat_of_code : np.ndarray, global_column -> feat_index (matches _one_hot_layout)
      n_codes      : total one-hot columns (== M.shape[1])

    Iterates feat_cols in order and sorted codes per feature -- bit-identical to
    _one_hot_layout's inner loop, just without building M."""
    col_of = {}
    feat_of_code = []
    for f_i, feat in enumerate(feat_cols):
        codes_seen = set()
        for v in atlas_sub[feat].dropna():
            codes_seen.update(v)
        for c in sorted(codes_seen):
            col_of[(f_i, c)] = len(feat_of_code)
            feat_of_code.append(f_i)
    return col_of, np.asarray(feat_of_code, dtype=np.int64), len(feat_of_code)


def encode_against_atlas(universe_wide, feat_cols, col_of, n_codes):
    """One-hot encode every universe language into rows of width n_codes,
    aligned to col_of. Codes a language attests that ATLAS never observed are
    dropped (they carry no training signal). Returns (M_universe, universe_langs).

    `universe_wide` must share columns with the ATLAS layout (both come from
    data_sources.combined_wide(sources)); pass wide.reindex(index=universe)."""
    universe_langs = list(universe_wide.index)
    M = np.zeros((len(universe_langs), n_codes), dtype=np.float32)
    for li, g in enumerate(universe_langs):
        for f_i, feat in enumerate(feat_cols):
            v = universe_wide.at[g, feat]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            for c in v:
                col = col_of.get((f_i, c))
                if col is not None:
                    M[li, col] = 1.0
    return M, universe_langs


def build_rf(floor=0.70, sources=("grambank", "wals")):
    """Fit the validated full_tuned RandomForest on the ATLAS modeling set (the
    languages with BTS ground truth that clear the coverage floor), and capture
    the fixed ATLAS one-hot column space so the universe can be encoded against
    it. Returns a bundle dict (see the plan's function inventory for keys)."""
    bts, langs_ids, langs, wide, scripts = fm._load_common(floor, sources)  # mode='gb'
    feat_cols = list(wide.columns)
    atlas_sub = wide.reindex(index=langs_ids)
    M, foc, has, feats, idx = _grambank._one_hot_layout(atlas_sub, feat_cols)
    col_of, _, n_codes = feature_code_columns(atlas_sub, feat_cols)
    pairs = fm.directed_pairs(langs_ids)
    bts_idx = bts.set_index(["glottocode_s", "glottocode_t"])
    y = np.array([float(bts_idx.loc[(s, t), "bts"]) for s, t in pairs])
    X2 = fm.stage2_design(pairs, {g: k for k, g in enumerate(idx)}, M, foc, len(feats))
    rf = RandomForestRegressor(
        n_estimators=200, random_state=42,
        **fm.rf_kwargs("full_tuned", sources)).fit(X2, y)
    return {
        "rf": rf, "feat_cols": feat_cols, "feat_of_code": foc, "n_feat": len(feats),
        "col_of": col_of, "n_codes": n_codes, "atlas_ids": set(langs_ids),
        "wide": wide, "scripts": scripts, "langs": langs, "bts": bts,
    }


def atlas_affinity(universe_langs, atlas_ids, wide):
    """{glottocode: max simple-similarity to any ATLAS training language}.

    An ATLAS language saturates to 1.0 (its self-similarity on the diagonal); a
    non-ATLAS language is < 1.0. This is the RF-extrapolation confidence basis:
    how well the ATLAS training distribution covers the language."""
    members = list(dict.fromkeys(
        [g for g in universe_langs if g in wide.index]
        + [g for g in atlas_ids if g in wide.index]))
    sim = _grambank.simple_similarity(_grambank.subset_wide(wide, members))
    atlas_in = [a for a in atlas_ids if a in sim.columns]
    out = {}
    for g in universe_langs:
        if g not in sim.index:
            out[g] = 0.0
        elif g in atlas_ids:
            out[g] = 1.0
        else:
            out[g] = float(sim.loc[g, atlas_in].max()) if atlas_in else 0.0
    return out


def _route_pair(s, t, affinity, atlas_ids, route, auto_tau):
    """Which model serves pair (s, t)? 'rf' or 'generalizer'."""
    if route == "rf":
        return "rf"
    if route == "generalizer":
        return "rf" if (s in atlas_ids and t in atlas_ids) else "generalizer"
    if route == "auto":
        return ("rf" if (affinity.get(s, 0.0) >= auto_tau
                         and affinity.get(t, 0.0) >= auto_tau) else "generalizer")
    raise ValueError(f"unknown route {route!r}; expected 'rf', 'generalizer', or 'auto'")


def _fit_generalizer(bundle):
    """Ridge on [similarity, genealogical, geographic] fit on the ATLAS training
    pairs -- the source-comparable fallback (mirrors predict_transfer._gen)."""
    atlas = sorted(bundle["atlas_ids"])
    wide, langs, scripts = bundle["wide"], bundle["langs"], bundle["scripts"]
    pairs = fm.directed_pairs(atlas)
    bts_idx = bundle["bts"].set_index(["glottocode_s", "glottocode_t"])
    y = np.array([float(bts_idx.loc[(s, t), "bts"]) for s, t in pairs])
    sim = _grambank.simple_similarity(_grambank.subset_wide(wide, atlas))
    gene = _grambank.genealogical_distance(langs, atlas)
    geo = _grambank.geographic_distance(langs, atlas)
    Xg = fm.stage1_design(pairs, sim, gene, geo, scripts)[:, [0, 2, 3]]
    return Ridge(alpha=1.0).fit(Xg, y)


def _generalizer_predict(gen, s, t, sim, gene, geo):
    """Predict BTS for (s, t) from [sim, gene, geo], or None if a language lacks
    genealogical/geographic data (e.g. a WALS-only language absent from Grambank)."""
    try:
        x = np.array([[float(sim.loc[s, t]), float(gene.loc[s, t]), float(geo.loc[s, t])]])
    except KeyError:
        return None
    if not np.isfinite(x).all():
        return None
    return float(gen.predict(x)[0])


def _chunked_rf_predict(rf, pairs, pos, M, foc, n_feat):
    """RF predictions for a list of pairs, encoded in memory-bounded chunks."""
    out = np.empty(len(pairs), dtype=np.float64)
    for start in range(0, len(pairs), CHUNK):
        chunk = pairs[start:start + CHUNK]
        X = fm.stage2_design(chunk, pos, M, foc, n_feat)
        out[start:start + len(chunk)] = rf.predict(X)
    return out


def build_matrix(universe, bundle, route="rf", auto_tau=0.5):
    """Long-format prediction matrix for every directed pair in `universe`.

    Columns: source, target, bts, model_served, is_extrapolation, confidence.
    Routing per `--route`; confidence is sqrt(atlas_affinity_s * atlas_affinity_t)
    for RF cells and sqrt(coverage_s * coverage_t) for generalizer cells. Pairs a
    language with no features is absent from are dropped up front."""
    atlas_ids = bundle["atlas_ids"]
    wide = bundle["wide"]
    universe = [g for g in universe if g in wide.index]
    pairs = fm.directed_pairs(universe)
    affinity = atlas_affinity(universe, atlas_ids, wide)
    cov = atlas_languages.compute_coverage(wide)

    want_rf = [i for i, (s, t) in enumerate(pairs)
               if _route_pair(s, t, affinity, atlas_ids, route, auto_tau) == "rf"]
    rf_set = set(want_rf)
    want_gen = [i for i in range(len(pairs)) if i not in rf_set]

    bts = np.full(len(pairs), np.nan)
    served = ["none"] * len(pairs)
    conf = np.zeros(len(pairs))

    if want_rf:
        M_univ, ulangs = encode_against_atlas(
            wide.reindex(index=universe), bundle["feat_cols"],
            bundle["col_of"], bundle["n_codes"])
        pos_univ = {g: i for i, g in enumerate(ulangs)}
        rf_pred = _chunked_rf_predict(
            bundle["rf"], [pairs[i] for i in want_rf], pos_univ, M_univ,
            bundle["feat_of_code"], bundle["n_feat"])
        for j, i in enumerate(want_rf):
            s, t = pairs[i]
            bts[i] = rf_pred[j]
            served[i] = "rf"
            conf[i] = float(np.sqrt(max(affinity[s] * affinity[t], 0.0)))

    if want_gen:
        gen = _fit_generalizer(bundle)
        members = list(dict.fromkeys(universe + [g for g in atlas_ids if g in wide.index]))
        sub = _grambank.subset_wide(wide, members)
        sim = _grambank.simple_similarity(sub)
        gene = _grambank.genealogical_distance(bundle["langs"], list(sub.index))
        geo = _grambank.geographic_distance(bundle["langs"], list(sub.index))
        for i in want_gen:
            s, t = pairs[i]
            p = _generalizer_predict(gen, s, t, sim, gene, geo)
            if p is None:
                served[i] = "none"
                conf[i] = 0.0
            else:
                served[i] = "generalizer"
                bts[i] = p
                conf[i] = float(np.sqrt(max(float(cov.get(s, 0.0)) * float(cov.get(t, 0.0)), 0.0)))

    extrap = [bool((s not in atlas_ids) or (t not in atlas_ids)) for s, t in pairs]
    return pd.DataFrame({
        "source": [str(s) for s, _ in pairs],
        "target": [str(t) for _, t in pairs],
        "bts": bts,
        "model_served": served,
        "is_extrapolation": extrap,
        "confidence": conf,
    })


def _family_of(g, langs):
    """Top-level Glottolog family for a glottocode (first token of lineage), or
    '' if unknown -- used only to group the heatmap axes."""
    if langs is None:
        return ""
    row = langs[langs["ID"] == g]
    if row.empty:
        return ""
    lin = str(row.iloc[0].get("lineage", "") or "")
    toks = [t for t in lin.split("/") if t]
    return toks[0] if toks else ""


def _plot_heatmap(mat, universe, png_path, langs=None):
    """Minimal Figure C.2-style heatmap (RdBu_r, family-ordered axes where the
    Glottolog lineage is available, else glottocode order)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = list(universe)
    if langs is not None:
        order = sorted(universe, key=lambda g: (_family_of(g, langs), g))
    pos = {g: i for i, g in enumerate(universe)}
    oi = [pos[g] for g in order]
    M = np.asarray(mat, dtype=float)[np.ix_(oi, oi)]
    finite = np.isfinite(M)
    vmin = float(M[finite].min()) if finite.any() else -1.0
    vmax = float(M[finite].max()) if finite.any() else 1.0

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 0.3), max(5, len(order) * 0.3)))
    cmap = matplotlib.colormaps.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d9d9d9")
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, aspect="equal",
              vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=5, rotation=90)
    ax.set_yticklabels(order, fontsize=5)
    ax.set_xlabel("source")
    ax.set_ylabel("target")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def write_outputs(long_df, universe, label, out_dir="outputs", plot=True, langs=None):
    """Write long CSV (always), wide BTS grid, wide confidence grid, and a
    heatmap PNG when len(universe) <= 60. Returns {kind: Path}."""
    out = Path(out_dir) / f"transfer_matrix__{label}"
    out.mkdir(parents=True, exist_ok=True)

    long_path = out / f"transfer_matrix_long__{label}.csv"
    long_df.to_csv(long_path, index=False)

    # cell[target, source] = bts(source->target) -- matches plot_transfer_matrices
    bw = long_df.pivot(index="target", columns="source", values="bts")
    bw = bw.reindex(index=universe, columns=universe)
    wide_path = out / f"transfer_matrix__{label}.csv"
    bw.to_csv(wide_path)

    cw = long_df.pivot(index="target", columns="source", values="confidence")
    cw = cw.reindex(index=universe, columns=universe)
    conf_path = out / f"transfer_matrix_confidence__{label}.csv"
    cw.to_csv(conf_path)

    paths = {"long": long_path, "wide": wide_path, "confidence": conf_path}
    if plot and len(universe) <= 60:
        png_path = out / f"transfer_matrix__{label}.png"
        _plot_heatmap(bw.to_numpy(), universe, png_path, langs=langs)
        paths["png"] = png_path
    return paths


def _preset_set(name):
    if name == "atlas24":
        return set(atlas_languages.atlas_glottocodes(0.70))
    if name == "atlas28":
        return set(atlas_languages.atlas_glottocodes(0.0))
    raise ValueError(f"unknown preset {name!r}; expected 'atlas24' or 'atlas28'")


def _parse_languages(arg):
    """Allowlist from a comma string OR a CSV/text file with a 'glottocode'
    column (or one glottocode per line)."""
    import csv
    p = Path(arg)
    if p.exists():
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            fields = [n.lower() for n in (reader.fieldnames or [])]
            if "glottocode" in fields:
                return {row["glottocode"].strip() for row in reader
                        if (row.get("glottocode") or "").strip()}
        with open(p) as f:
            return {line.strip() for line in f if line.strip()}
    return {x.strip() for x in arg.split(",") if x.strip()}


def _print_summary(long_df, universe, atlas_ids, label, paths):
    counts = long_df["model_served"].value_counts().to_dict()
    n_extrap = int(long_df["is_extrapolation"].sum())
    mean_conf = float(long_df["confidence"].mean())
    print(f"[{label}] universe: {len(universe)} languages ({len(atlas_ids)} ATLAS training)")
    print(f"cells by model: {counts} | extrapolation: {n_extrap}/{len(long_df)} "
          f"| mean confidence: {mean_conf:.3f}")
    print("NOTE: RF predictions beyond the ATLAS training set are unvalidated "
          "(no BTS ground truth); weight off-distribution cells by 'confidence'.")
    for kind, p in paths.items():
        print(f"  wrote {kind}: {p}")


def main():
    """CLI: resolve a universe, fit the RF, predict the matrix, write outputs."""
    import argparse
    ap = argparse.ArgumentParser(
        description="Predict the BTS transfer matrix for an open universe of languages.")
    ap.add_argument("--sources", default="grambank,wals",
                    help="comma-separated CLDF sources (default: grambank,wals)")
    ap.add_argument("--floor", type=float, default=0.70,
                    help="RF training coverage floor (default 0.70 -> ATLAS-24)")
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="universe coverage floor (default 0.70; lower to open the universe)")
    ap.add_argument("--lang-set", choices=["gb", "union", "intersect"], default="gb",
                    help="universe eligibility mode (gb=Grambank floor, union=GB OR WALS, "
                         "intersect=GB AND WALS)")
    ap.add_argument("--route", choices=["rf", "generalizer", "auto"], default="rf",
                    help="per-cell routing (rf=RF everywhere; generalizer=RF in-set only; "
                         "auto=RF in-distribution, generalizer far out)")
    ap.add_argument("--auto-tau", type=float, default=0.5,
                    help="affinity threshold for --route auto (default 0.5)")
    ap.add_argument("--preset", choices=["atlas24", "atlas28"], default=None)
    ap.add_argument("--languages", default=None,
                    help="allowlist: comma-separated glottocodes or a CSV/text file path")
    ap.add_argument("--exclude", default=None, help="comma-separated glottocodes to drop")
    ap.add_argument("--label", default=None, help="output run label (default: auto)")
    ap.add_argument("--out", default="outputs", help="output directory (default: outputs)")
    ap.add_argument("--no-plot", action="store_true", help="skip the heatmap PNG")
    a = ap.parse_args()

    sources = tuple(s.strip() for s in a.sources.split(","))
    coverage = per_source_coverage(sources)
    wide = data_sources.combined_wide(sources)
    preset = _preset_set(a.preset) if a.preset else None
    languages = _parse_languages(a.languages) if a.languages else None
    exclude = set(x.strip() for x in a.exclude.split(",")) if a.exclude else None
    universe = resolve_universe(coverage, list(wide.index), a.lang_set, a.threshold,
                                preset=preset, languages=languages, exclude=exclude)
    if len(universe) < 2:
        raise SystemExit(f"universe has {len(universe)} language(s); need >= 2")

    bundle = build_rf(a.floor, sources)
    long_df = build_matrix(universe, bundle, route=a.route, auto_tau=a.auto_tau)
    label = a.label or f"{a.route}_{a.lang_set}_t{int(round(a.threshold * 100))}_n{len(universe)}"
    paths = write_outputs(long_df, universe, label, out_dir=a.out,
                          plot=not a.no_plot, langs=bundle["langs"])
    _print_summary(long_df, universe, bundle["atlas_ids"], label, paths)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
