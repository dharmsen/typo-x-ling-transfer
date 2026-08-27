#!/usr/bin/env python3
"""Download WALS CLDF data into ./data/wals/.

Clones github.com/cldf-datasets/wals (shallow, branch master), copies the four
core CLDF CSVs (values/parameters/languages/codes) from cldf/ into ./data/wals/.
Skips if the expected files are already present. Mirrors
grambank_similarity/download_data.py.

Fallback if the full clone is too heavy: fetch just the four CSVs via
https://raw.githubusercontent.com/cldf-datasets/wals/master/cldf/<file>.csv
with urllib.request.urlretrieve (stdlib, no new dependency).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/cldf-datasets/wals.git"
BRANCH = "master"
EXPECTED_FILES = ("values.csv", "parameters.csv", "languages.csv", "codes.csv")

HERE = Path(__file__).resolve().parents[1]
DATA_DIR = HERE / "data" / "wals"
TMP_CLONE = HERE / "data" / ".wals_clone"


def _find_cldf_dir(clone_root: Path) -> Path:
    """Locate the dir holding values.csv (cldf/ at top level, else glob)."""
    cldf = clone_root / "cldf"
    if (cldf / "values.csv").exists():
        return cldf
    candidates = list(clone_root.glob("**/values.csv"))
    if candidates:
        src = candidates[0].parent
        print(f"  cldf/ not at top level; using {src.relative_to(clone_root)}")
        return src
    raise FileNotFoundError(f"no values.csv found under {clone_root}")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    missing = [f for f in EXPECTED_FILES if not (DATA_DIR / f).exists()]
    if not missing:
        print(f"All {len(EXPECTED_FILES)} files already present in {DATA_DIR}; skipping.")
        return 0

    if TMP_CLONE.exists():
        shutil.rmtree(TMP_CLONE)

    print(f"Cloning {REPO_URL} (shallow, {BRANCH}) ...")
    rc = subprocess.call(
        ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(TMP_CLONE)])
    if rc != 0:
        print(f"ERROR: git clone failed (rc={rc})", file=sys.stderr)
        return rc

    try:
        try:
            src = _find_cldf_dir(TMP_CLONE)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        copied = 0
        for name in EXPECTED_FILES:
            s, d = src / name, DATA_DIR / name
            if s.exists():
                shutil.copy2(s, d)
                copied += 1
                print(f"  copied {name} ({s.stat().st_size:,} bytes)")
            else:
                print(f"  WARNING: {name} missing in source", file=sys.stderr)
    finally:
        shutil.rmtree(TMP_CLONE, ignore_errors=True)

    print(f"Done. {copied}/{len(EXPECTED_FILES)} files in {DATA_DIR}")
    return 0 if copied == len(EXPECTED_FILES) else 2


if __name__ == "__main__":
    sys.exit(main())
