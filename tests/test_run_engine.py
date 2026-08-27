import os
import stat
import textwrap
from pathlib import Path
import pytest
import run as R


def _pkg(tmp_path):
    (tmp_path / "configs" / "experiments").mkdir(parents=True)
    (tmp_path / "outputs" / "logs").mkdir(parents=True)
    (tmp_path / "references").mkdir(parents=True)
    (tmp_path / "data").mkdir()
    return tmp_path


def _script(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def test_run_step_success_env_cwd_log(tmp_path, capsys):
    _pkg(tmp_path)
    _script(tmp_path, "hello.py", """
        import os, pathlib
        print("GB=" + os.environ.get("GRAMBANK_DATA", "?"))
        print("WALS=" + os.environ.get("WALS_DATA", "?"))
        print("cwd=" + os.getcwd())
        pathlib.Path("out.csv").write_text("ok")
    """)
    step = {"id": "s", "command": ["hello.py"], "outputs": ["out.csv"]}
    dt = R.run_step(step, tmp_path)
    assert dt >= 0
    log = (tmp_path / "outputs" / "logs" / "s.log").read_text()
    assert "GB=" + str(tmp_path / "data" / "grambank") in log
    assert "cwd=" + str(tmp_path) in log
    assert (tmp_path / "out.csv").exists()


def test_run_step_missing_output_exits(tmp_path):
    _pkg(tmp_path)
    _script(tmp_path, "noop.py", "print('hi')\n")
    step = {"id": "s", "command": ["noop.py"], "outputs": ["out.csv"]}
    with pytest.raises(SystemExit):
        R.run_step(step, tmp_path)


def test_run_step_nonzero_rc_exits(tmp_path):
    _pkg(tmp_path)
    _script(tmp_path, "bad.py", "import sys; sys.exit(3)\n")
    step = {"id": "s", "command": ["bad.py"]}
    with pytest.raises(SystemExit):
        R.run_step(step, tmp_path)


def test_ensure_data_runs_fetcher(tmp_path):
    _pkg(tmp_path)
    _script(tmp_path, "fetch.py", """
        import pathlib, sys
        pathlib.Path(sys.argv[1]).write_text("x")
    """)
    markers = {"demo": tmp_path / "data" / "marker.txt"}
    fetchers = {"demo": ["fetch.py", str(tmp_path / "data" / "marker.txt")]}
    R.ensure_data(["demo"], tmp_path, fetchers=fetchers, markers=markers)
    assert markers["demo"].exists()


def test_ensure_data_unknown_dep_exits(tmp_path):
    with pytest.raises(SystemExit):
        R.ensure_data(["nope"], tmp_path)


def test_verify_step_against_reference(tmp_path):
    _pkg(tmp_path)
    (tmp_path / "out.csv").write_text("a\n0.5\n")
    (tmp_path / "references" / "out.csv").write_text("a\n0.5\n")
    step = {"id": "s", "verify": ["out.csv"]}
    ok, msg = R.verify_step(step, tmp_path)
    assert ok, msg


def test_verify_step_flags_drift(tmp_path):
    _pkg(tmp_path)
    (tmp_path / "out.csv").write_text("a\n0.9\n")
    (tmp_path / "references" / "out.csv").write_text("a\n0.5\n")
    step = {"id": "s", "verify": ["out.csv"]}
    ok, _ = R.verify_step(step, tmp_path)
    assert not ok
