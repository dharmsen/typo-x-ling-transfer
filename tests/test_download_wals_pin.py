# tests for the WALS pin marker logic of data_setup/download_wals.py: the
# fetcher must skip only when the files exist AND the recorded pin matches,
# so upstream drift cannot silently change the CLDF inputs.
from data_setup import download_wals as dw


def test_marker_matches_true_on_recorded_pin(tmp_path):
    (tmp_path / dw.MARKER_NAME).write_text(dw.WALS_COMMIT + "\n")
    assert dw.marker_matches(tmp_path) is True


def test_marker_matches_false_when_missing(tmp_path):
    assert dw.marker_matches(tmp_path) is False


def test_marker_matches_false_on_other_pin(tmp_path):
    (tmp_path / dw.MARKER_NAME).write_text("deadbeef\n")
    assert dw.marker_matches(tmp_path) is False


def test_pin_is_a_full_hex_sha():
    assert len(dw.WALS_COMMIT) == 40
    int(dw.WALS_COMMIT, 16)  # raises if not hex
