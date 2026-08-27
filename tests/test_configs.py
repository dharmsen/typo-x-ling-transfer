from pathlib import Path
import run as R

STEPS = R.load_steps()
PRESETS = R.load_presets()


def test_all_preset_ids_exist():
    for pid, ids in PRESETS.items():
        unknown = [i for i in ids if i not in STEPS]
        assert not unknown, f"preset {pid}: {unknown}"


def test_every_step_runnable_somewhere():
    used = {i for ids in PRESETS.values() for i in ids}
    orphan = set(STEPS) - used
    assert not orphan, f"steps in no preset: {orphan}"


def test_prereq_outputs_declared():
    for sid, s in STEPS.items():
        for p in s.get("prerequisites", []):
            assert STEPS[p].get("outputs"), f"{sid} -> {p} declares no outputs"


def test_verify_tables_have_references():
    pkg = Path(__file__).resolve().parents[1]
    for sid, s in STEPS.items():
        for rel in s.get("verify", []):
            assert (pkg / "references" / Path(rel).name).exists(), \
                f"{sid} verifies {rel} but references/ lacks it"


def test_data_deps_known():
    for sid, s in STEPS.items():
        for d in s.get("data", []):
            assert d in R.DATA_MARKERS, f"{sid}: unknown data dep {d}"
