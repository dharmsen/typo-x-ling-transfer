import textwrap
from pathlib import Path
import pandas as pd
import pytest
import run as R


def _step_dir(tmp_path, bodies):
    d = tmp_path / "experiments"
    d.mkdir()
    for name, body in bodies.items():
        (d / f"{name}.yaml").write_text(textwrap.dedent(body))
    return d


def test_load_steps_reads_id_and_command(tmp_path):
    d = _step_dir(tmp_path, {"a": "id: a\ndescription: t\ncommand: [x.py]\n"})
    steps = R.load_steps(d)
    assert steps["a"]["command"] == ["x.py"]


def test_load_steps_requires_id(tmp_path):
    d = _step_dir(tmp_path, {"a": "description: t\ncommand: [x.py]\n"})
    with pytest.raises(SystemExit):
        R.load_steps(d)


def test_resolve_ids_preset_and_step(tmp_path):
    steps = {"a": {"id": "a"}, "b": {"id": "b"}}
    presets = {"p": ["a", "b"]}
    assert R.resolve_ids("a,b", None, steps, presets) == ["a", "b"]
    assert R.resolve_ids(None, "p", steps, presets) == ["a", "b"]
    with pytest.raises(SystemExit):
        R.resolve_ids(None, "q", steps, presets)
    with pytest.raises(SystemExit):
        R.resolve_ids("nope", None, steps, presets)


def test_check_prereqs_passes_when_run_earlier(tmp_path):
    steps = {"a": {"id": "a", "command": ["a.py"], "outputs": ["out.csv"]},
             "b": {"id": "b", "command": ["b.py"], "prerequisites": ["a"]}}
    R.check_prereqs(["a", "b"], steps, tmp_path)  # no exit


def test_check_prereqs_exits_when_output_missing(tmp_path):
    steps = {"a": {"id": "a", "command": ["a.py"], "outputs": ["out.csv"]},
             "b": {"id": "b", "command": ["b.py"], "prerequisites": ["a"]}}
    with pytest.raises(SystemExit):
        R.check_prereqs(["b"], steps, tmp_path)


def test_check_prereqs_passes_when_output_on_disk(tmp_path):
    (tmp_path / "out.csv").write_text("x")
    steps = {"a": {"id": "a", "command": ["a.py"], "outputs": ["out.csv"]},
             "b": {"id": "b", "command": ["b.py"], "prerequisites": ["a"]}}
    R.check_prereqs(["b"], steps, tmp_path)


def _write_pair(tmp_path, rows_new, rows_ref):
    new, ref = tmp_path / "new.csv", tmp_path / "ref.csv"
    pd.DataFrame(rows_new).to_csv(new, index=False)
    pd.DataFrame(rows_ref).to_csv(ref, index=False)
    return new, ref


def test_compare_table_identical(tmp_path):
    new, ref = _write_pair(tmp_path, {"a": [1.0, 2.0], "b": ["x", "y"]},
                           {"a": [1.0, 2.0], "b": ["x", "y"]})
    ok, msg = R.compare_table(new, ref)
    assert ok, msg


def test_compare_table_float_tolerance(tmp_path):
    new, ref = _write_pair(tmp_path, {"a": [1.0000005]}, {"a": [1.0]})
    ok, _ = R.compare_table(new, ref)
    assert ok


def test_compare_table_flags_numeric_drift(tmp_path):
    new, ref = _write_pair(tmp_path, {"a": [0.7]}, {"a": [0.5]})
    ok, msg = R.compare_table(new, ref)
    assert not ok and "a" in msg


def test_compare_table_nan_matches_nan(tmp_path):
    # e.g. ood_taxonomy.csv: TRUE_EXTRAP row is NaN in both tables (the
    # reference stores an empty field, the regeneration a literal NaN)
    new, ref = _write_pair(tmp_path, {"a": [0.7, float("nan")]},
                           {"a": [0.7, float("nan")]})
    ok, msg = R.compare_table(new, ref)
    assert ok, msg


def test_compare_table_flags_nan_vs_number(tmp_path):
    new, ref = _write_pair(tmp_path, {"a": [float("nan")]}, {"a": [0.7]})
    ok, _ = R.compare_table(new, ref)
    assert not ok


def test_compare_table_flags_structure(tmp_path):
    new, ref = _write_pair(tmp_path, {"a": [1], "b": [2]}, {"a": [1]})
    ok, msg = R.compare_table(new, ref)
    assert not ok


BIAS_LOG = """feature, std_weight, mean_abs_contrib_BTS
script_diff, -0.0770, 0.0710
log1p(resource_s), +0.0870, 0.0700

R2(B,y) = 0.120
"""


def test_parse_bias_terms_log():
    rows = R.parse_bias_terms_log(BIAS_LOG)
    assert rows["script_diff"] == (-0.077, 0.071)
    assert rows["log1p(resource_s)"] == (0.087, 0.070)
