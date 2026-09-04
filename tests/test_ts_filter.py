"""Tests for the global tilt-series filter (square_aperture_montage.ts_filter).

Covers the two matching modes: anchored name matching for the tilt-series steps
(blend, make-mdoc) and substring matching for the file-based steps (crop, fill),
plus wildcard support and the pattern-normalisation helper.
"""

from square_aperture_montage.ts_filter import (
    normalize_patterns,
    ts_matches,
    path_matches,
    filter_names,
    filter_paths,
)


# Realistic filenames from SerialEM square-aperture data.
TS_NAMES = [
    "VLP3x3_p01_ts_002",
    "VLP3x3_p01_ts_003",
    "VLP3x3_p04_ts_004",
]
# Crop-input average filename: ts name embedded behind an acquisition timestamp.
CROP_FILE = ("frames/averages/2025-02-17_21.40.45_VLP3x3_p04_ts_004_"
             "VLP3x3_p04_ts_004_0_0_001_0000_-0.0.mrc")
FILL_FILE = "blended/frames/VLP3x3_p01_ts_002_-52.0_blended_frames.mrc"


# ---------------------------------------------------------------------------
# normalize_patterns
# ---------------------------------------------------------------------------

def test_normalize_patterns():
    assert normalize_patterns(None) == []
    assert normalize_patterns([]) == []
    assert normalize_patterns("") == []
    assert normalize_patterns("VLP3x3_p01_ts_002") == ["VLP3x3_p01_ts_002"]
    assert normalize_patterns(["a", "b"]) == ["a", "b"]
    assert normalize_patterns([2, 3]) == ["2", "3"]


# ---------------------------------------------------------------------------
# ts_matches — anchored (blend / make-mdoc)
# ---------------------------------------------------------------------------

def test_empty_patterns_match_everything():
    assert ts_matches("VLP3x3_p01_ts_002", []) is True
    assert path_matches("anything.mrc", []) is True


def test_ts_matches_exact_backward_compatible():
    assert ts_matches("VLP3x3_p01_ts_002", ["VLP3x3_p01_ts_002"]) is True
    assert ts_matches("VLP3x3_p01_ts_003", ["VLP3x3_p01_ts_002"]) is False


def test_ts_matches_wildcard():
    assert ts_matches("VLP3x3_p01_ts_002", ["VLP3x3_p01_ts_*"]) is True
    assert ts_matches("VLP3x3_p04_ts_004", ["VLP3x3_p01_ts_*"]) is False
    assert ts_matches("VLP3x3_p04_ts_004", ["*ts_004"]) is True


def test_ts_matches_anchored_not_substring():
    # Anchored: a bare fragment does NOT match unless wildcarded.
    assert ts_matches("VLP3x3_p01_ts_002", ["ts_002"]) is False
    assert ts_matches("VLP3x3_p01_ts_002", ["*ts_002"]) is True


def test_filter_names():
    assert filter_names(TS_NAMES, ["VLP3x3_p01_ts_*"]) == [
        "VLP3x3_p01_ts_002", "VLP3x3_p01_ts_003",
    ]
    assert filter_names(TS_NAMES, []) == TS_NAMES
    assert filter_names(TS_NAMES, ["VLP3x3_p04_ts_004"]) == ["VLP3x3_p04_ts_004"]


# ---------------------------------------------------------------------------
# path_matches — substring (crop / fill)
# ---------------------------------------------------------------------------

def test_path_matches_embedded_name():
    # Plain ts name matches even though it is not a filename prefix.
    assert path_matches(CROP_FILE, ["VLP3x3_p04_ts_004"]) is True
    assert path_matches(CROP_FILE, ["VLP3x3_p01_ts_002"]) is False


def test_path_matches_prefix_and_wildcard():
    assert path_matches(FILL_FILE, ["VLP3x3_p01_ts_002"]) is True
    assert path_matches(CROP_FILE, ["VLP3x3_p04_ts_*"]) is True
    assert path_matches(FILL_FILE, ["VLP3x3_p04_ts_*"]) is False


def test_filter_paths():
    files = [CROP_FILE, FILL_FILE]
    assert filter_paths(files, ["VLP3x3_p04_ts_004"]) == [CROP_FILE]
    assert filter_paths(files, ["VLP3x3_p01_ts_002"]) == [FILL_FILE]
    assert filter_paths(files, []) == files
    assert filter_paths(files, ["VLP3x3_p01_ts_*", "VLP3x3_p04_ts_*"]) == files
