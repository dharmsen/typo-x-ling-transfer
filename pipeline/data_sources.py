# data_sources.py
"""Multi-source CLDF typology data layer for the BLT proxy.

A small source registry: each CLDF source (Grambank, WALS, ...) is described by
where its data lives (env-overridable) and how to fold its rows onto the shared
Glottocode index. combined_wide(sources=[...]) returns one Glottocode-keyed
wide matrix (Language x Feature -> frozenset|NaN) with source-prefixed feature
IDs, ready for the (already column-generic) _grambank math.

Grambank is already Glottocode-indexed (its Language_ID IS the Glottocode), so
it needs no re-key. WALS keys rows by WALS code; we re-key to Glottocode via
languages.csv and drop rows lacking one.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import _grambank
from pipeline import paths

HERE = Path(__file__).resolve().parents[1]


def wals_dir() -> Path:
    """Resolve the WALS CLDF dir (env-overridable). Raise if missing."""
    d = Path(os.environ.get("WALS_DATA", HERE / "data" / "wals"))
    if not d.is_dir():
        raise FileNotFoundError(
            f"WALS CLDF not found at {d}. Run `uv run download_wals.py` first, "
            f"or set the WALS_DATA env var."
        )
    return d


def _wals_glottocode_map(wals_dir: Path) -> dict[str, str]:
    """{WALS code (languages.ID) -> Glottocode}; entries with no Glottocode omitted."""
    langs = pd.read_csv(wals_dir / "languages.csv", dtype=str,
                        usecols=["ID", "Glottocode"])
    out: dict[str, str] = {}
    for wals_id, glotto in zip(langs["ID"], langs["Glottocode"]):
        if isinstance(glotto, str) and glotto.strip():
            out[str(wals_id)] = glotto.strip()
    return out


def _rekey_to_glottocode(wide: pd.DataFrame, id2glotto: dict[str, str]) -> pd.DataFrame:
    """Re-index `wide` from source Language_ID onto Glottocode.

    Rows with no Glottocode mapping are dropped (count printed). Multiple
    source IDs mapping to one Glottocode are unioned per feature (frozenset
    union; np.nan where every member lacks data)."""
    keep = [i for i in wide.index if i in id2glotto]
    dropped = len(wide.index) - len(keep)
    if dropped:
        print(f"  [{dropped} source rows lacked a Glottocode and were dropped]",
              file=sys.stderr)
    sub = wide.loc[keep]
    new_index = [id2glotto[i] for i in sub.index]
    feats = list(sub.columns)
    arr = sub.to_numpy()
    # group row positions by target Glottocode
    groups: dict[str, list[int]] = {}
    for pos, g in enumerate(new_index):
        groups.setdefault(g, []).append(pos)
    rows, idx = [], []
    for g, positions in groups.items():
        row = {}
        for f_i, feat in enumerate(feats):
            sets = [arr[p, f_i] for p in positions]
            sets = [s for s in sets if isinstance(s, frozenset)]
            row[feat] = frozenset().union(*sets) if sets else np.nan
        rows.append(row)
        idx.append(g)
    out = pd.DataFrame(rows, index=idx, columns=feats)
    out.index.name = wide.index.name
    return out


# Re-key is None for Grambank (already Glottocode-indexed); WALS is re-keyed
# onto Glottocode via _wals_glottocode_map + _rekey_to_glottocode.
SOURCES: dict[str, dict] = {
    "grambank": {"dir": paths.require_grambank_data, "prefix": "", "rekey": None},
    "wals": {"dir": wals_dir, "prefix": "WALS_", "rekey": _wals_glottocode_map},
}


DEFAULT_FLOOR = 0.70


def sources_suffix(sources, mode="gb", floor=DEFAULT_FLOOR) -> str:
    """'' for the Grambank-only baseline, else '__' + '_'.join(sources).

    Appends '_union'/'_intersect' when `mode` selects a per-source language
    set (atlas_languages.modeling_glottocodes), and '_fNN' (e.g. '_f50') when
    the coverage floor differs from the 0.70 default -- so union/intersect and
    floor-sweep result tables never clobber the default gb tables."""
    sources = tuple(sources)
    s = "" if sources == ("grambank",) else "__" + "_".join(sources)
    if mode in ("union", "intersect"):
        s += "_" + mode
    if abs(floor - DEFAULT_FLOOR) > 1e-9:
        s += "_f%d" % int(round(floor * 100))
    return s


def _load_source_wide(name: str) -> pd.DataFrame:
    """One source's wide matrix: Glottocode-indexed, feature IDs prefixed."""
    spec = SOURCES[name]
    values = pd.read_csv(spec["dir"]() / "values.csv", dtype=str)
    wide, _ = _grambank.build_wide_matrix(values)          # returns (wide, cell_sizes)
    if spec["prefix"]:
        wide = wide.rename(columns=lambda c: f"{spec['prefix']}{c}")
    if spec["rekey"] is not None:
        wide = _rekey_to_glottocode(wide, spec["rekey"](spec["dir"]()))
    return wide


def combined_wide(sources=("grambank", "wals")) -> pd.DataFrame:
    """Glottocode-keyed wide matrix combining the given CLDF sources.

    Outer-joins sources on the Glottocode index (column union). Feature IDs are
    source-prefixed (Grambank GBxxx, WALS WALS_*). Sorts rows and columns."""
    sources = tuple(sources)
    unknown = [s for s in sources if s not in SOURCES]
    if unknown:
        raise ValueError(f"unknown source(s) {unknown}; known: {list(SOURCES)}")
    mats = [_load_source_wide(s) for s in sources]
    for name, m in zip(sources, mats):
        if m.shape[1] == 0:
            raise ValueError(f"source {name!r} contributed 0 feature columns")
    combined = pd.concat(mats, axis=1) if mats else pd.DataFrame()
    return combined.sort_index(axis=0).sort_index(axis=1)


def load_wikipedia_articles(path=None):
    """{glottocode: article_count} from the committed snapshot. Rows without a
    glottocode (ATLAS languages absent from Grambank) are skipped."""
    path = Path(path) if path else (HERE / "data" / "wikipedia_articles.csv")
    df = pd.read_csv(path, dtype=str)
    out = {}
    for g, n in zip(df["glottocode"], df["articles"]):
        if isinstance(g, str) and g.strip():
            try:
                out[g.strip()] = int(float(n))
            except (TypeError, ValueError):
                continue
    return out


def wals_family_genus(wals_dir_path=None):
    """{glottocode: (family, genus)} from WALS, re-keyed to Glottocode. First
    WALS row per Glottocode wins. Provides the major/minor family framing."""
    d = Path(wals_dir_path) if wals_dir_path else wals_dir()
    langs = pd.read_csv(d / "languages.csv", dtype=str,
                        usecols=["ID", "Glottocode", "Family", "Genus"])
    id2g = _wals_glottocode_map(d)
    out = {}
    for wid, fam, gen in zip(langs["ID"], langs["Family"], langs["Genus"]):
        g = id2g.get(wid)
        if not g or g in out:
            continue
        if isinstance(fam, str) and isinstance(gen, str):
            out[g] = (fam, gen)
    return out
