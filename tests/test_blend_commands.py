"""Tests for the blendmont command strings produced by blend_tiles.

IMOD is not available in CI, so every ``subprocess.run`` call is monkeypatched
to a no-op stub and the tests assert on the *generated command strings* and the
returned ``commands`` list. The central concern is the ``frame_edge_reuse``
option: that ``none`` reproduces the historical behaviour byte-for-byte, and
that ``edges`` / ``edges-xcorr`` add exactly the right flags and ``cp`` lines.
"""

import pytest

from square_aperture_montage import blend_tiles


TS = "VLP3x3_p04_ts_004"
TILT = -60.0   # float, so generated filenames contain '.' and '-'


class _StubResult:
    """Minimal stand-in for subprocess.CompletedProcess."""
    returncode = 0
    stdout = b""
    stderr = b""


def _fake_run(cmd, *args, **kwargs):
    return _StubResult()


def _make_sections():
    """A simple 2x2 montage of tiles with a uniform pixel-shift grid."""
    grid = [(0, 0), (3840, 0), (0, 3840), (3840, 3840)]
    sections = []
    for i, (sx, sy) in enumerate(grid):
        sections.append({
            "TiltAngle": TILT,
            "PixelShiftFromCenter": [sx, sy],
            "SubFramePath": f"X:\\data\\tile_{i}.tif",
        })
    return sections


def _run_worker(tmp_path, monkeypatch, frame_edge_reuse, create_exts,
                num_frames=2, blend_frames=True):
    """Set up temp dirs, optionally pre-create averages edge files, and run
    ``_blend_tilt_worker`` with ``subprocess.run`` stubbed out.

    ``create_exts`` is the list of averages edge-file extensions to create in
    the averages processing dir before running (e.g. ['.xef', '.yef']).

    Returns the ``commands`` list produced by the worker.
    """
    monkeypatch.setattr(blend_tiles.subprocess, "run", _fake_run)

    proc_avg = tmp_path / "processing" / "blending_averages"
    proc_frm = tmp_path / "processing" / "blending_frames"
    out_avg = tmp_path / "blended" / "averages"
    out_frm = tmp_path / "blended" / "frames"
    crop_avg = tmp_path / "cropped" / "averages"
    crop_frm = tmp_path / "cropped" / "frames"
    for d in (proc_avg, proc_frm, out_avg, out_frm, crop_avg, crop_frm):
        d.mkdir(parents=True, exist_ok=True)

    # Pre-create the averages edge files the frame runs would copy. In real
    # runs these are written by the averages blendmont (here mocked out).
    avg_root = proc_avg / f"{TS}_{TILT}_blended"
    for ext in create_exts:
        (proc_avg / f"{avg_root.name}{ext}").write_text("stub edge data\n")

    args = (
        0,                       # tilt_i
        _make_sections(),        # tile_sections
        TS,
        str(crop_avg), str(crop_frm),
        str(proc_avg), str(proc_frm),
        str(out_avg), str(out_frm),
        11664,                   # blend_size
        blend_frames,
        num_frames,
        False, False,            # normalize averages / frames
        None,                    # log_dir
        True,                    # snap_shifts_to_grid
        frame_edge_reuse,
    )

    _tilt_i, _angle, _avg, _frm, commands = blend_tiles._blend_tilt_worker(args)
    return commands


def _blendmont(commands):
    return [c for c in commands if c.startswith("blendmont")]


def _frame_blendmont(commands):
    return [c for c in _blendmont(commands) if "frame" in c]


def _avg_blendmont(commands):
    return [c for c in _blendmont(commands) if "frame" not in c]


def _cp_lines(commands):
    return [c for c in commands if c.startswith("cp -f ")]


# ---------------------------------------------------------------------------
# frame_edge_reuse = 'none'  — regression guard
# ---------------------------------------------------------------------------

def test_none_has_no_reuse_flags(tmp_path, monkeypatch):
    commands = _run_worker(tmp_path, monkeypatch, "none", [])

    # Exactly one averages blendmont + one per frame.
    assert len(_avg_blendmont(commands)) == 1
    assert len(_frame_blendmont(commands)) == 2

    # No blendmont command carries a reuse flag, and each ends with the
    # historical '-adj -shift' tail (byte-for-byte unchanged from main).
    for cmd in _blendmont(commands):
        assert "-oldedge" not in cmd
        assert "-readxcorr" not in cmd
        assert cmd.endswith("-adj -shift")

    # No edge-file copies happen in 'none' mode.
    assert _cp_lines(commands) == []


# ---------------------------------------------------------------------------
# frame_edge_reuse = 'edges'
# ---------------------------------------------------------------------------

def test_edges_adds_oldedge_to_frames_only(tmp_path, monkeypatch):
    commands = _run_worker(tmp_path, monkeypatch, "edges", [".xef", ".yef"])

    # Averages call is untouched.
    for cmd in _avg_blendmont(commands):
        assert "-oldedge" not in cmd
        assert cmd.endswith("-adj -shift")

    # Every frame call gains -oldedge (and only that).
    for cmd in _frame_blendmont(commands):
        assert "-readxcorr" not in cmd
        assert cmd.endswith("-adj -shift -oldedge")

    # Two cp lines per frame (.xef, .yef), quoted.
    cps = _cp_lines(commands)
    assert len(cps) == 2 * 2
    assert all(c.count('"') == 4 for c in cps)
    assert any(c.endswith('.xef"') for c in cps)
    assert any(c.endswith('.yef"') for c in cps)
    assert not any(c.endswith('.ecd"') for c in cps)


# ---------------------------------------------------------------------------
# frame_edge_reuse = 'edges-xcorr'
# ---------------------------------------------------------------------------

def test_edges_xcorr_adds_both_flags(tmp_path, monkeypatch):
    commands = _run_worker(tmp_path, monkeypatch, "edges-xcorr",
                           [".xef", ".yef", ".ecd"])

    for cmd in _avg_blendmont(commands):
        assert cmd.endswith("-adj -shift")

    for cmd in _frame_blendmont(commands):
        assert cmd.endswith("-adj -shift -oldedge -readxcorr")

    # Three cp lines per frame (.xef, .yef, .ecd).
    cps = _cp_lines(commands)
    assert len(cps) == 3 * 2
    assert any(c.endswith('.ecd"') for c in cps)


# ---------------------------------------------------------------------------
# cp lines run in the right place (after newstack, before blendmont)
# ---------------------------------------------------------------------------

def test_cp_ordering(tmp_path, monkeypatch):
    commands = _run_worker(tmp_path, monkeypatch, "edges", [".xef", ".yef"])

    # For frame0: newstack -> cp .xef/.yef -> blendmont, in that order.
    ns_idx = next(i for i, c in enumerate(commands)
                  if c.startswith("newstack") and "frame0" in c)
    bm_idx = next(i for i, c in enumerate(commands)
                  if c.startswith("blendmont") and "frame0" in c)
    cp_idxs = [i for i, c in enumerate(commands)
               if c.startswith("cp -f ") and "frame0_blended" in c]

    assert cp_idxs, "expected cp lines for frame0"
    assert ns_idx < min(cp_idxs)
    assert max(cp_idxs) < bm_idx


# ---------------------------------------------------------------------------
# missing edge file -> warn and fall back, never raise
# ---------------------------------------------------------------------------

def test_missing_edge_file_falls_back(tmp_path, monkeypatch, capsys):
    # Request 'edges' but create no averages edge files at all.
    commands = _run_worker(tmp_path, monkeypatch, "edges", [])

    out = capsys.readouterr().out
    assert "[WARNING]" in out
    assert "from scratch" in out

    # Falls back to 'none' for every frame: no reuse flags, no cp lines.
    for cmd in _frame_blendmont(commands):
        assert "-oldedge" not in cmd
        assert cmd.endswith("-adj -shift")
    assert _cp_lines(commands) == []


# ---------------------------------------------------------------------------
# invalid value is rejected by process_tilt_series
# ---------------------------------------------------------------------------

def test_invalid_frame_edge_reuse_raises(tmp_path):
    with pytest.raises(ValueError):
        blend_tiles.process_tilt_series(
            ts=TS, mdoc_dir=str(tmp_path), cropped_averages_dir=str(tmp_path),
            cropped_frames_dir=str(tmp_path),
            processing_averages_dir=str(tmp_path),
            processing_frames_dir=str(tmp_path),
            output_averages_dir=str(tmp_path), output_frames_dir=str(tmp_path),
            output_averages_mdoc_dir=str(tmp_path),
            output_frames_mdoc_dir=str(tmp_path),
            blend_size=11664, blend_frames=True, num_frames=2,
            frame_edge_reuse="bogus",
        )
