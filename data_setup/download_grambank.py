#!/usr/bin/env python3
"""Download Grambank v1.0.3 CLDF data into data/grambank/.

Clones github.com/grambank/grambank at tag v1.0.3 (shallow), copies cldf/*.csv
into data/grambank/. Skips work if the expected files are already present.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/grambank/grambank.git"
TAG = "v1.0.3"  # pin the release the reference tables were built from
CLDF_SUBDIR = "cldf"
EXPECTED_FILES = ("values.csv", "parameters.csv", "languages.csv", "codes.csv")

HERE = Path(__file__).resolve().parents[1]
DATA_DIR = HERE / "data" / "grambank"
TMP_CLONE = HERE / "data" / ".grambank_clone"


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    missing = [f for f in EXPECTED_FILES if not (DATA_DIR / f).exists()]
    if not missing:
        print(f"All {len(EXPECTED_FILES)} files already present in {DATA_DIR}; skipping download.")
        return 0

    if TMP_CLONE.exists():
        shutil.rmtree(TMP_CLONE)

    print(f"Cloning {REPO_URL} at tag {TAG} (shallow) ...")
    rc = subprocess.call(["git", "clone", "--depth", "1", "--branch", TAG,
                          REPO_URL, str(TMP_CLONE)])
    if rc != 0:
        print(f"ERROR: git clone failed (rc={rc})", file=sys.stderr)
        return rc

    src = TMP_CLONE / CLDF_SUBDIR
    if not src.is_dir():
        # Some releases put cldf one level deeper; try to locate it.
        candidates = list(TMP_CLONE.glob("**/values.csv"))
        if candidates:
            src = candidates[0].parent
            print(f"  cldf/ not at top level; using {src.relative_to(TMP_CLONE)}")
        else:
            print(f"ERROR: no cldf/ directory found in clone", file=sys.stderr)
            return 1

    copied = 0
    for name in EXPECTED_FILES:
        s = src / name
        d = DATA_DIR / name
        if s.exists():
            shutil.copy2(s, d)
            copied += 1
            print(f"  copied {name} ({s.stat().st_size:,} bytes)")
        else:
            print(f"  WARNING: {name} missing in source", file=sys.stderr)

    shutil.rmtree(TMP_CLONE, ignore_errors=True)
    print(f"Done. {copied}/{len(EXPECTED_FILES)} files in {DATA_DIR}")
    return 0 if copied == len(EXPECTED_FILES) else 2


if __name__ == "__main__":
    sys.exit(main())
