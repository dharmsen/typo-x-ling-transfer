#!/usr/bin/env python
"""Run one part of robustness_controls.py with canonical arguments."""
import sys as _sys
from pathlib import Path as _Path
_PKG_ROOT = str(_Path(__file__).resolve().parents[1])
if _PKG_ROOT not in _sys.path:
    _sys.path.insert(0, _PKG_ROOT)
import argparse
from experiments import robustness_controls as rc

PARTS = {
    "metadata_tune": lambda: rc.tune_metadata(),
    "composition": lambda: rc.composition(),
    "digitize": lambda: rc.digitize(B=100),
    "shared22": lambda: rc.shared22(),
    "langboot": lambda: rc.langboot(B=2000),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=sorted(PARTS), required=True)
    a = ap.parse_args()
    PARTS[a.part]()
