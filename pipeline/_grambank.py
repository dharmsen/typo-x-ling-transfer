# _grambank.py
"""Pure functions PORTED from the parent research codebase (analyze.py,
genealogical_mantel.py, geographic_mantel.py). Kept verbatim where possible so
the proxy's typology math is bit-identical to the tested parent codebase.
Changes: read the CLDF via paths.require_grambank_data(); rarity metric dropped
(YAGNI - we use simple matching only); Mantel imported verbatim (later task).
"""
import numpy as np
import pandas as pd

from pipeline import paths

RANDOM_SEED = 42
PAIRWISE_MIN_OVERLAP = 50


def load_data():
    d = paths.require_grambank_data()
    values = pd.read_csv(d / "values.csv", dtype=str)
    params = pd.read_csv(d / "parameters.csv", dtype=str)
    langs = pd.read_csv(d / "languages.csv", dtype=str)
    codes = pd.read_csv(d / "codes.csv", dtype=str)
    return values, params, langs, codes


def clean_value_cell(v):
    if v is None:
        return None
    if not isinstance(v, str):
        return None
    v = v.strip()
    if v == "" or v.lower() == "nan":
        return None
    if v == "?":
        return None
    v = v.strip("?")
    if v == "":
        return None
    parts = [p.strip().strip("?").strip() for p in v.replace(";", ",").split(",")]
    parts = [p for p in parts if p and p != "?"]
    if not parts:
        return None
    return frozenset(parts)


def build_wide_matrix(values: pd.DataFrame):
    """Return (wide, cell_sizes): Language x Feature -> frozenset | NaN."""
    df = values[["Language_ID", "Parameter_ID", "Value"]].copy()
    df["codes"] = df["Value"].apply(clean_value_cell)
    df = df.dropna(subset=["codes"])
    grouped = df.groupby(["Language_ID", "Parameter_ID"], sort=False)["codes"].apply(
        lambda s: frozenset().union(*s)
    )
    wide = grouped.unstack(level="Parameter_ID")
    wide = wide.sort_index(axis=0).sort_index(axis=1)
    cell_sizes = grouped.groupby("Language_ID").size().rename("cells_with_data")
    return wide, cell_sizes


def compute_coverage(wide: pd.DataFrame) -> pd.DataFrame:
    return (wide.notna().sum(axis=0) / len(wide)).rename("coverage").to_frame()


def _one_hot_layout(wide, feat_subset):
    sub = wide[feat_subset]
    lang_index = sub.index.to_numpy()
    feat_cols = sub.columns.tolist()
    n_lang, n_feat = sub.shape
    rows, feat_of_code = [], []
    has_data = np.zeros((n_lang, n_feat), dtype=bool)
    for f_i, feat in enumerate(feat_cols):
        col = sub[feat]
        codes_seen = set()
        for v in col.dropna():
            codes_seen.update(v)
        codes_sorted = sorted(codes_seen)
        block = np.zeros((n_lang, len(codes_sorted)), dtype=np.float32)
        for li, v in enumerate(col.to_numpy()):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            has_data[li, f_i] = True
            for c in v:
                block[li, codes_sorted.index(c)] = 1.0
        rows.append(block)
        feat_of_code.extend([f_i] * len(codes_sorted))
    M = np.concatenate(rows, axis=1) if rows else np.zeros((n_lang, 0), np.float32)
    return M, np.asarray(feat_of_code, dtype=np.int64), has_data, feat_cols, lang_index


def _agreement_per_feature(M, feat_of_code, n_feat):
    out = {}
    for f in range(n_feat):
        cols = np.where(feat_of_code == f)[0]
        if len(cols) == 0:
            out[f] = np.zeros((M.shape[0], M.shape[0]), dtype=bool)
            continue
        out[f] = (M[:, cols] @ M[:, cols].T) > 0
    return out


def compute_similarity_matrix_simple(wide, feat_subset, min_overlap=None):
    """Simple matching: # agreeing features / # comparable features. PORTED
    (rarity branch dropped). Returns (sim float32 [n,n], lang_index, comparable)."""
    M, feat_of_code, has_data, _, lang_index = _one_hot_layout(wide, feat_subset)
    n_lang, n_feat = has_data.shape
    agree = _agreement_per_feature(M, feat_of_code, n_feat)
    has = has_data.astype(np.float32)
    comparable = has @ has.T
    agree_count = np.zeros((n_lang, n_lang), dtype=np.float32)
    for f in range(n_feat):
        agree_count += agree[f].astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(comparable > 0, agree_count / comparable, 0.0).astype(np.float32)
    if min_overlap is not None:
        sim = np.where(comparable >= min_overlap, sim, np.nan).astype(np.float32)
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32), lang_index, comparable


def subset_wide(wide, glottocodes):
    """Reindex the wide matrix to a glottocode subset (rows only)."""
    return wide.reindex(index=[g for g in glottocodes if g in wide.index])


def simple_similarity(wide_subset, min_overlap=None):
    """Pairwise/simple similarity over a wide SUBSET -> DataFrame keyed by glottocode.

    min_overlap defaults to None (no masking): the ATLAS langs are well-attested
    (high coverage), so an evidence floor mainly injects NaN. Set it (e.g. 30)
    only if a coverage audit shows weak pairs.
    """
    feats = list(wide_subset.columns)
    sim, idx, _ = compute_similarity_matrix_simple(wide_subset, feats, min_overlap)
    return pd.DataFrame(sim, index=idx, columns=idx)


def lineage_to_path(value, lang_id):
    if value is None or (not isinstance(value, str)) or value.strip() == "":
        return [str(lang_id)]
    tokens = [t for t in value.split("/") if t]
    return tokens + [str(lang_id)]


def genealogical_distance_matrix(paths_list):
    """N x N MRCA distance in [0,1]: 1 - 2k/(len_i+len_j). PORTED verbatim."""
    n = len(paths_list)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    max_d = max(len(p) for p in paths_list)
    all_tokens = sorted({t for p in paths_list for t in p})
    rank = {t: i for i, t in enumerate(all_tokens)}
    r_sentinel = len(all_tokens)
    keys = np.empty((n, max_d), dtype=np.int64)
    for i, p in enumerate(paths_list):
        L = len(p)
        for d in range(max_d):
            keys[i, d] = rank[p[d]] if d < L else r_sentinel + i
    share = np.ones((n, n), dtype=bool)
    common = np.zeros((n, n), dtype=np.int32)
    for d in range(max_d):
        col = keys[:, d]
        share &= col[:, None] == col[None, :]
        common += share
    depth = np.array([len(p) for p in paths_list], dtype=np.int32)
    depth_sum = depth[:, None].astype(np.float64) + depth[None, :].astype(np.float64)
    dist = 1.0 - 2.0 * common / depth_sum
    np.fill_diagonal(dist, 0.0)
    return dist


def genealogical_distance(langs, glottocodes):
    """Genealogical distance DataFrame over a glottocode subset, aligned to input order."""
    glottocodes = [g for g in glottocodes if g in set(langs["ID"])]
    lin = dict(zip(langs["ID"].tolist(), langs["lineage"].tolist()))
    paths_list = [lineage_to_path(lin.get(g), g) for g in glottocodes]
    d = genealogical_distance_matrix(paths_list)
    return pd.DataFrame(d, index=glottocodes, columns=glottocodes)


EARTH_RADIUS_KM = 6371.0


def haversine_matrix(lat, lon):
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    lat1, lat2 = lat_r[:, None], lat_r[None, :]
    lon1, lon2 = lon_r[:, None], lon_r[None, :]
    a = (np.sin((lat2 - lat1) / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2)
    d = 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    missing = ~np.isfinite(lat) | ~np.isfinite(lon)
    d = np.where(missing[:, None] | missing[None, :], np.nan, d)
    np.fill_diagonal(d, 0.0)
    return (d + d.T) / 2.0


def geographic_distance(langs, glottocodes):
    sub = langs.set_index("ID").loc[[g for g in glottocodes if g in set(langs["ID"])]]
    lat = pd.to_numeric(sub["Latitude"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(sub["Longitude"], errors="coerce").to_numpy(dtype=float)
    d = haversine_matrix(lat, lon)
    return pd.DataFrame(d, index=sub.index, columns=sub.index)


def sim_to_distance(sim):
    d = 1.0 - np.asarray(sim, dtype=np.float64)
    np.fill_diagonal(d, 0.0)
    return d


def _mantel_r(a_flat, b_flat):
    a = np.asarray(a_flat, dtype=np.float64)
    b = np.asarray(b_flat, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a, b = a[mask], b[mask]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _permuted_r(a_m, b_flat_full, i_m, j_m, n, perm):
    lo = np.minimum(perm[i_m], perm[j_m])
    hi = np.maximum(perm[i_m], perm[j_m])
    lin = lo * n - lo * (lo + 1) // 2 + (hi - lo - 1)
    return _mantel_r(a_m, b_flat_full[lin])


def mantel_test(dist_typo, dist_geo, B, seed=42):
    """Simple permutation Mantel test. PORTED verbatim. Returns {r,z,p,n_pairs,B}."""
    n = dist_typo.shape[0]
    i, j = np.triu_indices(n, k=1)
    a = dist_typo[i, j].astype(np.float64)
    b_flat_full = dist_geo[i, j].astype(np.float64)
    mask = np.isfinite(a) & np.isfinite(b_flat_full)
    i_m, j_m = i[mask], j[mask]
    a_m = a[mask]
    r_obs = _mantel_r(a_m, b_flat_full[mask])
    rng = np.random.default_rng(seed)
    r_perm = np.array([_permuted_r(a_m, b_flat_full, i_m, j_m, n, rng.permutation(n))
                       for _ in range(int(B))], dtype=np.float64)
    if not np.isfinite(r_obs):
        p, z = float("nan"), float("nan")
    else:
        fp = r_perm[np.isfinite(r_perm)]
        p = (int(np.sum(fp >= r_obs)) + 1) / (int(B) + 1)
        z = float((r_obs - fp.mean()) / fp.std()) if fp.size and fp.std() > 0 else float("nan")
    return {"r": r_obs, "z": z, "p": p, "n_pairs": int(mask.sum()), "B": int(B)}
