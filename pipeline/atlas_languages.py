# atlas_languages.py
"""ATLAS language -> Grambank glottocode mapping, coverage, and floor filtering."""
from pathlib import Path
import pandas as pd
from pipeline import _grambank

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
COVERAGE_FLOOR = 0.70  # high default; raise (stricter) if results are sparse/disappointing


def load_mapping() -> pd.DataFrame:
    return pd.read_csv(DATA / "atlas_languages.csv", dtype=str)


def compute_coverage(wide: pd.DataFrame) -> pd.Series:
    """Per-language fraction of features present."""
    return (wide.notna().sum(axis=1) / wide.shape[1]).rename("coverage")


def filter_by_floor(coverage: pd.Series, floor: float = COVERAGE_FLOOR) -> pd.Series:
    return coverage[coverage >= floor].sort_values(ascending=False)


def annotate_mapping(mapping: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Fill coverage column from the wide matrix."""
    cov = compute_coverage(wide)
    out = mapping.copy()
    out["coverage"] = out["glottocode"].map(cov).round(4)
    return out


def select_glottocodes(mapping: pd.DataFrame, floor: float = COVERAGE_FLOOR) -> list[str]:
    """Glottocodes (in mapping row order) whose coverage >= floor. Skips rows
    with empty/NaN glottocode (ABSENT languages) or non-numeric coverage.
    Iterates rows so duplicate or empty glottocodes cannot cause a crash."""
    out = []
    for g, c in zip(mapping["glottocode"], mapping["coverage"]):
        if not isinstance(g, str) or g.strip() == "":
            continue
        try:
            cv = float(c)
        except (TypeError, ValueError):
            continue
        if cv >= floor:
            out.append(g)
    return out


def atlas_glottocodes(floor: float = COVERAGE_FLOOR) -> list[str]:
    """Glottocodes of ATLAS languages meeting the coverage floor (modeling set)."""
    values, _, langs, _ = _grambank.load_data()
    wide, _ = _grambank.build_wide_matrix(values)
    mapping = annotate_mapping(load_mapping(), wide)
    return select_glottocodes(mapping, floor)


def _per_source_coverage(sources):
    """{source: pd.Series} of per-glottocode coverage over that source's wide
    matrix. A language absent from a source is simply missing from its Series
    (treated as coverage 0.0 by _select_by_mode)."""
    from pipeline import data_sources
    out = {}
    for s in sources:
        w = data_sources._load_source_wide(s)
        out[s] = w.notna().sum(axis=1) / w.shape[1]
    return out


def _select_by_mode(per_source_cov, atlas_glottocodes, floor, mode):
    """Pure OR/AND filter over per-source coverage Series.

    union: keep if ANY source covers the glottocode >= floor.
    intersect: keep if ALL sources do. A glottocode absent from a source counts
    as coverage 0.0 (fails the floor), so a language WALS doesn't Glottocode-map
    fails the intersection but can pass the union via Grambank."""
    keep = []
    for g in atlas_glottocodes:
        passes = [float(per_source_cov[s].get(g, 0.0)) >= floor for s in per_source_cov]
        if (mode == "union" and any(passes)) or (mode == "intersect" and all(passes)):
            keep.append(g)
    return keep


def modeling_glottocodes(floor: float = COVERAGE_FLOOR,
                         sources=("grambank", "wals"), mode="gb") -> list[str]:
    """ATLAS modeling set under a per-source coverage floor.

    mode='gb' (default): Grambank-only coverage >= floor -- bit-identical to
        atlas_glottocodes(floor); WALS is not consulted (the historic behavior,
        where the bar is GB-only and WALS only adds feature columns).
    mode='union': in if ANY configured source covers the glottocode >= floor.
        At the 0.70 GB+WALS default this is n=25 (the 24 GB-floor langs + Thai,
        which Grambank under-attests at 0.52 but WALS covers at 0.72).
    mode='intersect': in if ALL configured sources cover it >= floor (n=12 at
        the 0.70 GB+WALS default -- the well-attested-in-both core).

    Always a subset of the ATLAS mapping; fit_models._load_common further
    intersects with the BTS ground truth and the combined feature index."""
    if mode == "gb":
        return atlas_glottocodes(floor)
    if mode not in ("union", "intersect"):
        raise ValueError(f"unknown mode {mode!r}; expected 'gb', 'union', or 'intersect'")
    mapping = load_mapping()
    atlas_g = [g for g in mapping["glottocode"] if isinstance(g, str) and g.strip()]
    return _select_by_mode(_per_source_coverage(sources), atlas_g, floor, mode)
