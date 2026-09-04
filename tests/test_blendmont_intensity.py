"""Tests for the blendmont intensity-correction options.

These exercise the pure builder (:func:`build_intensity_args`), the config
reader (:func:`intensity_args_from_config`), and the wiring into
:func:`imod_blendmont`. No IMOD binaries and no MRC files are required — the
single subprocess call is monkeypatched so we can inspect the generated
command string.

The overriding invariant: with no intensity options set, the generated
blendmont command is byte-for-byte identical to the pre-existing one, so this
feature is strictly opt-in.
"""

import inspect
import os

import pytest

from square_aperture_montage.blend_tiles import (
    build_intensity_args,
    intensity_args_from_config,
    imod_blendmont,
    process_tilt_series,
    _blend_tilt_worker,
)
import square_aperture_montage.blend_tiles as bt


# ---------------------------------------------------------------------------
# subprocess.run stub — records every command, returns a success result
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self):
        self.returncode = 0
        self.stdout = b""
        self.stderr = b""


@pytest.fixture
def captured_cmds(monkeypatch):
    cmds = []

    def fake_run(cmd, *args, **kwargs):
        cmds.append(cmd)
        return _FakeResult()

    monkeypatch.setattr(bt.subprocess, "run", fake_run)
    return cmds


# ---------------------------------------------------------------------------
# 1. Defaults are a no-op — empty list and unchanged command string
# ---------------------------------------------------------------------------

def test_defaults_return_empty():
    assert build_intensity_args() == []


def test_default_command_is_unchanged(tmp_path, captured_cmds):
    proc = str(tmp_path)
    blended = os.path.join(proc, "TS_-52.0_blended.mrc")
    imod_blendmont("stack.mrc", "shifts.plin", "out.plout", 11664,
                   blended, proc)

    intermediate = os.path.join(proc, "TS_-52.0_blended_raw.mrc")
    roo = os.path.join(proc, "TS_-52.0_blended")
    expected = (
        f"blendmont -imin stack.mrc -plin shifts.plin "
        f"-imout {intermediate} "
        f"-roo {roo} "
        f"-al out.plout -adj -shift"
    )
    assert captured_cmds[0] == expected


# ---------------------------------------------------------------------------
# 2. -intensity value validation
# ---------------------------------------------------------------------------

def test_fix_from_edges_one():
    assert build_intensity_args(fix_from_edges=1) == ["-intensity", "1"]


def test_fix_from_edges_two():
    assert build_intensity_args(fix_from_edges=2) == ["-intensity", "2"]


def test_fix_from_edges_invalid_raises():
    with pytest.raises(ValueError):
        build_intensity_args(fix_from_edges=3)


# ---------------------------------------------------------------------------
# 3. -base ordering and formatting
# ---------------------------------------------------------------------------

def test_base_with_intensity_order():
    args = build_intensity_args(fix_from_edges=1, base=-32768)
    assert args == ["-intensity", "1", "-base", "-32768"]
    # -base must come immediately after -intensity
    assert args.index("-base") == args.index("-intensity") + 2


# ---------------------------------------------------------------------------
# 4. -sum and -other are mutually exclusive
# ---------------------------------------------------------------------------

def test_sum_and_other_mutually_exclusive():
    with pytest.raises(ValueError) as exc:
        build_intensity_args(fix_from_edges=1, sum_for_gradient=True,
                             other_gradient_file="grad.txt")
    msg = str(exc.value)
    assert "sum_for_gradient" in msg
    assert "other_gradient_file" in msg


# ---------------------------------------------------------------------------
# 5. Paths with spaces are shell-quoted
# ---------------------------------------------------------------------------

def test_paths_with_spaces_are_quoted():
    args = build_intensity_args(flatfield_file="my flat field.mrc")
    assert "-flatfield" in args
    quoted = args[args.index("-flatfield") + 1]
    assert quoted == "'my flat field.mrc'"


# ---------------------------------------------------------------------------
# 6. Config reader resolves paths against data_dir and validates existence
# ---------------------------------------------------------------------------

def test_config_resolves_relative_path(tmp_path):
    grad = tmp_path / "grad.txt"
    grad.write_text("gradient\n")
    args = intensity_args_from_config(
        {"fix_from_edges": 1, "other_gradient_file": "grad.txt"},
        str(tmp_path),
    )
    assert "-other" in args
    resolved = args[args.index("-other") + 1]
    # shlex.quote leaves a path with no special chars unquoted
    assert resolved == str(grad)


def test_config_missing_flatfield_raises(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        intensity_args_from_config(
            {"flatfield_file": "does_not_exist.mrc"},
            str(tmp_path),
        )
    assert "flatfield_file" in str(exc.value)


def test_empty_config_defaults_to_intensity_one():
    # fix_from_edges defaults to 1 when the block is absent or null.
    assert intensity_args_from_config({}, ".") == ["-intensity", "1"]
    assert intensity_args_from_config(None, ".") == ["-intensity", "1"]


def test_explicit_zero_disables_intensity():
    # An explicit 0 must still turn it off (not collapse into the default).
    assert intensity_args_from_config({"fix_from_edges": 0}, ".") == []


# ---------------------------------------------------------------------------
# 7. imod_blendmont appends the flags exactly once, after -shift
# ---------------------------------------------------------------------------

def test_imod_blendmont_appends_flags(tmp_path, captured_cmds):
    proc = str(tmp_path)
    blended = os.path.join(proc, "TS_0.0_blended.mrc")
    args = ["-intensity", "2", "-base", "0"]
    imod_blendmont("stack.mrc", "shifts.plin", "out.plout", 11664,
                   blended, proc, intensity_args=args)

    blend_cmd = captured_cmds[0]
    assert blend_cmd.count("-intensity") == 1
    assert blend_cmd.count("-base") == 1
    assert blend_cmd.endswith("-shift -intensity 2 -base 0")
    # flags come after -shift, not before
    assert blend_cmd.index("-shift") < blend_cmd.index("-intensity")


# ---------------------------------------------------------------------------
# 7b. The worker reports how many stale edge files it cleared (for the summary)
# ---------------------------------------------------------------------------

def test_worker_reports_edge_removed_count(tmp_path, captured_cmds):
    proc_avg = tmp_path / "proc_avg"
    proc_avg.mkdir()
    (tmp_path / "out_avg").mkdir()
    ts = "TSX"
    # Rootname the averages blendmont will use: {ts}_{tilt_angle}_blended
    for ext in ("ecd", "xef", "yef"):
        (proc_avg / f"{ts}_0.0_blended.{ext}").write_text("")

    sections = [
        {"TiltAngle": 0.0, "PixelShiftFromCenter": [0, 0], "SubFramePath": "a.tif"},
        {"TiltAngle": 0.0, "PixelShiftFromCenter": [3682, 0], "SubFramePath": "b.tif"},
    ]
    args = (
        0, sections,
        ts, str(tmp_path), str(tmp_path),
        str(proc_avg), str(tmp_path / "proc_frm"),
        str(tmp_path / "out_avg"), str(tmp_path / "out_frm"),
        11664, False, 0,               # blend_size, blend_frames, num_frames
        False, False,                  # normalize avgs / frames
        None, True,                    # log_dir, snap_shifts_to_grid
        ("-intensity", "1"),           # intensity_args
    )
    result = _blend_tilt_worker(args)
    edge_removed = result[5]
    assert edge_removed == 3


# ---------------------------------------------------------------------------
# 8. Opt-in regression — the intensity_args default is empty everywhere
# ---------------------------------------------------------------------------

def test_intensity_args_default_is_empty():
    assert (inspect.signature(process_tilt_series)
            .parameters["intensity_args"].default == ())
    assert (inspect.signature(imod_blendmont)
            .parameters["intensity_args"].default == ())
