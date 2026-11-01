# checkpoint.py
"""Append-only JSONL checkpoints for the long nested runs.

Each completed work unit is appended as one JSON line and flushed, so an
interrupt (Ctrl-C, kill, power loss) loses at most the units still in flight;
a restart loads the file and skips every unit already present. A torn final
line (the process died mid-append) is dropped on load; corruption anywhere
before the final line raises, because that is a real error rather than an
interrupt.
"""
import json
from pathlib import Path


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    records = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break  # torn final line from an interrupted append
            raise ValueError(f"corrupt checkpoint line {i + 1} in {path}") from None
    return records
