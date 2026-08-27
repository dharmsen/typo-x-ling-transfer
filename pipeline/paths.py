# paths.py
"""Single source of truth for the Grambank CLDF location.

The CLDF lives at ./data/grambank, fetched by data_setup/download_grambank.py; the GRAMBANK_DATA env var still overrides.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
_DEFAULT = HERE / "data" / "grambank"


def grambank_data_dir() -> Path:
    return Path(os.environ.get("GRAMBANK_DATA", _DEFAULT))


def require_grambank_data() -> Path:
    """Return the dir or raise with a helpful message (incl. fallback command)."""
    d = grambank_data_dir()
    if not d.is_dir():
        raise FileNotFoundError(
            f"Grambank CLDF not found at {d}. Run "
            f"`uv run data_setup/download_grambank.py` first, or set "
            f"the GRAMBANK_DATA env var."
        )
    return d
