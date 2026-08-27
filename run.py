#!/usr/bin/env python3
"""Single entry point regenerating every experiment of the paper.

Steps are declared in configs/experiments/*.yaml; presets in
configs/presets.yaml. Every step runs as a subprocess with the package root
as cwd and data roots pointed at ./data, fetching missing data first.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
STEPS_DIR = HERE / "configs" / "experiments"
PRESETS_PATH = HERE / "configs" / "presets.yaml"
DATA = HERE / "data"
LOGS = HERE / "outputs" / "logs"
REFERENCES = HERE / "references"

NUMERIC_TOL = {"rtol": 1e-3, "atol": 1e-3}

FETCHERS = {
    "grambank": ["data_setup/download_grambank.py"],
    "wals": ["data_setup/download_wals.py"],
    "wikipedia_counts": ["data_setup/download_wikipedia.py"],
    "wikipedia_text": ["data_setup/download_wikipedia_text.py", "--articles", "2000",
                       "--modeling-only"],
}
DATA_MARKERS = {
    "grambank": DATA / "grambank" / "values.csv",
    "wals": DATA / "wals" / "values.csv",
    "wikipedia_counts": DATA / "wikipedia_articles.csv",
    "wikipedia_text": DATA / "wikipedia_text",
}


def load_steps(steps_dir: Path = STEPS_DIR) -> dict:
    steps = {}
    for p in sorted(steps_dir.glob("*.yaml")):
        cfg = yaml.safe_load(p.read_text())
        if not cfg or "id" not in cfg:
            sys.exit(f"{p} is missing the 'id' field")
        steps[cfg["id"]] = cfg
    return steps


def load_presets(path: Path = PRESETS_PATH) -> dict:
    return yaml.safe_load(path.read_text())["presets"]


def resolve_ids(arg_steps, arg_preset, steps, presets):
    if not (arg_steps or arg_preset):
        sys.exit("give --preset or --step (see --list)")
    if arg_preset:
        if arg_preset not in presets:
            sys.exit(f"unknown preset '{arg_preset}'; have {sorted(presets)}")
        ids = list(presets[arg_preset])
    else:
        ids = [s.strip() for s in arg_steps.split(",") if s.strip()]
    unknown = [i for i in ids if i not in steps]
    if unknown:
        sys.exit(f"unknown step ids: {unknown}")
    return ids


def check_prereqs(ids, steps, base: Path) -> None:
    done = set()
    for i in ids:
        for p in steps[i].get("prerequisites", []):
            outs = steps[p].get("outputs", [])
            on_disk = any((base / o).exists() for o in outs)
            if p not in done and not on_disk:
                sys.exit(f"step '{i}' needs prerequisite '{p}' "
                         f"({' '.join(steps[p]['command'])}); run it first or "
                         f"order it before '{i}' in the preset")
        done.add(i)


def compare_table(new: Path, ref: Path, rtol=1e-3, atol=1e-3):
    a, b = pd.read_csv(new), pd.read_csv(ref)
    if list(a.columns) != list(b.columns):
        return False, "columns differ"
    if len(a) != len(b):
        return False, f"row count {len(a)} vs {len(b)}"
    for col in a.columns:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            dx = x.astype(float) - y.astype(float)
            import numpy as np
            if not np.allclose(x.astype(float), y.astype(float), rtol=rtol,
                               atol=atol, equal_nan=True):
                return False, f"{col}: max |diff| {float(dx.abs().max()):.6g}"
        else:
            if not x.astype(str).equals(y.astype(str)):
                return False, f"{col}: values differ"
    return True, "ok"


def parse_bias_terms_log(text: str) -> dict:
    rows = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_()]+), ([+-]?\d+\.\d+), (\d+\.\d+)$", line.strip())
        if m:
            rows[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return rows


def ensure_data(deps, base: Path, fetchers=None, markers=None) -> None:
    fetchers = fetchers or FETCHERS
    markers = markers or DATA_MARKERS
    for d in deps:
        if d not in markers:
            sys.exit(f"unknown data dep '{d}'")
        if markers[d].exists():
            continue
        if d not in fetchers:
            sys.exit(f"no fetcher for data dep '{d}'")
        cmd = [sys.executable, *fetchers[d]]
        print(f"[data] {d} missing -> {' '.join(fetchers[d])}")
        rc = subprocess.call(cmd, cwd=base)
        if rc != 0 or not markers[d].exists():
            sys.exit(f"fetcher for '{d}' failed (rc={rc})")


def run_step(step: dict, base: Path = HERE) -> float:
    logs = base / "outputs" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ,
           "GRAMBANK_DATA": str(base / "data" / "grambank"),
           "WALS_DATA": str(base / "data" / "wals")}
    env["PYTHONPATH"] = (str(base) + os.pathsep + env["PYTHONPATH"]
                         if env.get("PYTHONPATH") else str(base))
    cmd = [sys.executable, *step["command"]]
    log_path = logs / f"{step['id']}.log"
    print(f"[run ] {step['id']}: {' '.join(step['command'])}")
    t0 = time.time()
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=base, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fh.write(line)
        rc = proc.wait()
    dt = time.time() - t0
    if rc != 0:
        sys.exit(f"step '{step['id']}' failed rc={rc}; see {log_path}")
    missing = [o for o in step.get("outputs", []) if not (base / o).exists()]
    if missing:
        sys.exit(f"step '{step['id']}' produced no {missing}; see {log_path}")
    print(f"[done] {step['id']} in {dt:.0f}s")
    return dt


BIAS_TERMS_EXPECTED = {
    "script_diff": (-0.077, 0.071),
    "genealogical_dist": (-0.034, 0.027),
    "geo_dist_norm": (-0.018, 0.016),
    "log1p(resource_s)": (0.087, 0.070),
    "log1p(resource_t)": (0.002, 0.002),
}


def verify_step(step: dict, base: Path = HERE):
    for rel in step.get("verify", []):
        new, ref = base / rel, base / "references" / Path(rel).name
        if not new.exists():
            return False, f"missing output {rel}"
        if not ref.exists():
            return False, f"no reference table for {rel}"
        ok, msg = compare_table(new, ref)
        if not ok:
            return False, f"{rel}: {msg}"
    if step.get("verify_stdout") == "bias_terms":
        log = (base / "outputs" / "logs" / f"{step['id']}.log").read_text()
        rows = parse_bias_terms_log(log)
        if set(rows) != set(BIAS_TERMS_EXPECTED):
            return False, f"parsed bias rows {sorted(rows)}"
        for k, (w, c) in BIAS_TERMS_EXPECTED.items():
            ew, ec = rows[k]
            if abs(w - ew) > 5e-3 or abs(c - ec) > 5e-3:
                return False, f"bias term {k}: {rows[k]} vs {(w, c)}"
    return True, "ok"


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset")
    ap.add_argument("--step", help="comma-separated step ids")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    steps = load_steps()
    presets = load_presets()
    if a.list:
        for pid, ids in presets.items():
            print(f"{pid}: {' '.join(ids)}")
        print("steps: " + " ".join(sorted(steps)))
        return
    ids = resolve_ids(a.step, a.preset, steps, presets)
    if a.dry_run:
        for i in ids:
            print(f"{i}: {' '.join(steps[i]['command'])}")
        print("(dry run: nothing is executed; prerequisites are checked at run time)")
        return
    check_prereqs(ids, steps, HERE)
    failures = []
    for i in ids:
        s = steps[i]
        ensure_data(s.get("data", []), HERE)
        run_step(s, HERE)
        if a.verify:
            if not s.get("verify") and not s.get("verify_stdout"):
                print(f"[verify] {i}: SKIP (no comparison declared)")
                continue
            ok, msg = verify_step(s, HERE)
            print(f"[verify] {i}: {'PASS' if ok else 'FAIL ' + msg}")
            if not ok:
                failures.append(i)
    if failures:
        sys.exit(f"verification failed for: {failures}")


if __name__ == "__main__":
    main()
