#!/usr/bin/env python3
"""Download WALS CLDF data into ./data/wals/.

Clones github.com/cldf-datasets/wals, checks out the pinned commit WALS_COMMIT,
and copies the four core CLDF CSVs (values/parameters/languages/codes) from
cldf/ into ./data/wals/, recording the pin in data/wals/.pinned_commit. Skips
only when the files are present AND the recorded pin matches; a drifted clone
is deleted and re-downloaded at the pin.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/cldf-datasets/wals.git"
WALS_COMMIT = "81298aa26d01fdc11b769bcc7fa87ad9e43d7ccd"  # v2020.6 pin
EXPECTED_FILES = ("values.csv", "parameters.csv", "languages.csv", "codes.csv")
MARKER_NAME = ".pinned_commit"

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


def marker_matches(data_dir: Path = DATA_DIR) -> bool:
    """True iff data_dir/.pinned_commit records the current WALS_COMMIT pin."""
    marker = data_dir / MARKER_NAME
    return marker.exists() and marker.read_text().strip() == WALS_COMMIT


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    missing = [f for f in EXPECTED_FILES if not (DATA_DIR / f).exists()]
    if not missing and marker_matches():
        print(f"All {len(EXPECTED_FILES)} files already present in {DATA_DIR} "
              f"at pin {WALS_COMMIT[:8]}; skipping.")
        return 0
    if not missing:
        marker = DATA_DIR / MARKER_NAME
        recorded = marker.read_text().strip() if marker.exists() else "<none>"
        print(f"PIN DRIFT: {DATA_DIR} holds no/other pin (marker: {recorded}, "
              f"want {WALS_COMMIT}); deleting and re-downloading at the pin")
        shutil.rmtree(DATA_DIR)
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TMP_CLONE.exists():
        shutil.rmtree(TMP_CLONE)

    print(f"Cloning {REPO_URL} (pinned at {WALS_COMMIT[:8]}) ...")
    rc = subprocess.call(["git", "clone", REPO_URL, str(TMP_CLONE)])
    if rc != 0:
        print(f"ERROR: git clone failed (rc={rc})", file=sys.stderr)
        return rc
    rc = subprocess.call(
        ["git", "-C", str(TMP_CLONE), "checkout", WALS_COMMIT])
    if rc != 0:
        print(f"ERROR: git checkout {WALS_COMMIT} failed (rc={rc})", file=sys.stderr)
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

    if copied == len(EXPECTED_FILES):
        (DATA_DIR / MARKER_NAME).write_text(WALS_COMMIT + "\n")
    print(f"Done. {copied}/{len(EXPECTED_FILES)} files in {DATA_DIR} "
          f"(pin {WALS_COMMIT[:8]})")
    return 0 if copied == len(EXPECTED_FILES) else 2


if __name__ == "__main__":
    sys.exit(main())
