# tests for the JSONL checkpoint helper used by the nested (leak-free) runs.
import json

import pytest

from pipeline import checkpoint as cp


def test_append_then_load_roundtrip(tmp_path):
    p = tmp_path / "ck.jsonl"
    cp.append_jsonl(p, {"fold": 0, "val": 1.5})
    cp.append_jsonl(p, {"fold": 1, "val": 2.5})
    recs = cp.load_jsonl(p)
    assert recs == [{"fold": 0, "val": 1.5}, {"fold": 1, "val": 2.5}]


def test_load_skips_torn_final_line(tmp_path):
    """A hard kill mid-append can truncate the last line; load must drop it."""
    p = tmp_path / "ck.jsonl"
    p.write_text('{"fold": 0, "val": 1.5}\n{"fold": 1, "val": 2.5}\n{"fold": 2, "va')
    recs = cp.load_jsonl(p)
    assert [r["fold"] for r in recs] == [0, 1]


def test_load_raises_on_corrupt_middle_line(tmp_path):
    """Corruption before the final line is a real error, not an interrupt."""
    p = tmp_path / "ck.jsonl"
    p.write_text('{"fold": 0}\n{"fold": BROKEN}\n{"fold": 2}\n')
    with pytest.raises(ValueError):
        cp.load_jsonl(p)


def test_load_missing_file_returns_empty(tmp_path):
    assert cp.load_jsonl(tmp_path / "nope.jsonl") == []
