# config.py
"""YAML experiment configs — the source of truth for the recommender (and going
forward, other experiments). load_config() -> ExperimentConfig; CLI --set k=v overrides apply on
top; write_meta() drops a _meta.json beside each result for reproducibility.

NOTE: deliberately does NOT use `from __future__ import annotations` —
apply_overrides inspects real type objects on dataclass fields to coerce CLI
strings (int / float / bool / comma-split tuple).
"""
import json
import subprocess
import types
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path


@dataclass
class ExperimentConfig:
    # output identity
    label: str = "default"
    out_dir: str = "outputs"
    # data / modeling
    floor: float = 0.70
    lang_set: str = "gb"
    sources: tuple = ()                 # candidate source glottocodes
    targets: tuple = ()                 # target glottocodes
    # NOTE: tuple-typed fields MUST stay bare `tuple`, NOT `tuple[str, ...]` —
    # _coerce matches on the raw type to comma-split --set overrides; a
    # types.GenericAlias would silently fail that match.
    sources_file: str | None = None
    targets_file: str | None = None
    # recommender
    matrix: str | None = None
    score_col: str = "bts"
    id_scheme: str = "glottocode"       # 'glottocode' (proxy/debiased) | 'atlas' (ground truth)
    objective: str = "transfer_max"     # transfer_max | coverage_greedy | centrality
    budget_k: int | None = None
    regime: str = "both"                # a1 | a2 | both
    coverage_threshold: float = 0.70
    on_low_coverage: str = "exclude"    # exclude | flag | allow
    # stats (used by Plan 2; harmless here)
    seeds: tuple = (42,)
    B_null: int = 1000
    B_boot: int = 2000


def _coerce(field_obj, raw):
    """Coerce a CLI string to the dataclass field's type."""
    if raw is None:
        return None
    t = field_obj.type
    # Peel X | None unions down to the single concrete type.
    if isinstance(t, types.UnionType):
        concrete = [a for a in t.__args__ if a is not type(None)]
        if len(concrete) == 1:
            t = concrete[0]
    if t in (tuple, list):
        return tuple(x.strip() for x in str(raw).split(",") if x.strip())
    if t is int:
        return int(raw)
    if t is float:
        return float(raw)
    if t is bool:
        return str(raw).lower() in ("1", "true", "yes")
    return raw  # str passthrough


def load_config(path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file. Unknown keys are ignored so a
    shared config can carry fields other experiments use. YAML lists are coerced
    to tuples for tuple-typed fields so the dataclass stays internally consistent
    (sources/targets round-trip as tuples, matching apply_overrides)."""
    import yaml
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    by_name = {f.name: f for f in fields(ExperimentConfig)}
    coerced = {}
    for k, v in data.items():
        if k not in by_name:
            continue
        if by_name[k].type in (tuple, list) and isinstance(v, list):
            coerced[k] = tuple(v)
        else:
            coerced[k] = v
    return ExperimentConfig(**coerced)


def apply_overrides(cfg: ExperimentConfig, overrides) -> ExperimentConfig:
    """Apply ['key=val', ...] CLI overrides (type-coerced) onto a copy."""
    by_name = {f.name: f for f in fields(ExperimentConfig)}
    new = replace(cfg)
    for item in overrides or []:
        key, _, val = item.partition("=")
        if key not in by_name:
            raise ValueError(
                "unknown config key %r; known: %s" % (key, sorted(by_name)))
        setattr(new, key, _coerce(by_name[key], val))
    return new


def write_meta(cfg: ExperimentConfig, path, extra=None) -> Path:
    """Write _meta.json with the resolved config + git commit + extra fields."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unknown"
    payload = {"config": asdict(cfg), "git_commit": commit}
    if extra:
        payload.update(extra)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path
