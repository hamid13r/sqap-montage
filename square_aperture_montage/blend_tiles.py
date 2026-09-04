#!/usr/bin/env python3
"""
blend_tiles.py — Blend 3×3 (or N×M) tilt-series montage tiles into a single tilt-series.

For each tilt-series discovered in the mdoc directory this script:
  1. Stacks the cropped average images for each tilt angle using IMOD newstack
  2. Blends the stack using IMOD blendmont (pixel shifts read from the mdoc)
  3. Resizes the blended result with IMOD clip
  4. Optionally repeats steps 1-3 frame-by-frame for per-frame stacks
  5. Writes an output .mdoc file pointing at the blended images

Typical usage
-------------
  sam-blend

  sam-blend --mdoc-dir Mdoc --averages-dir cropped/averages --blend-size 11664

Run ``sam-blend --help`` for all options.
"""

import concurrent.futures
import glob
import logging
import os
import shlex
import sys
from pathlib import Path, PureWindowsPath
import subprocess

import click
import tqdm

from .mdoc_reader import parse_mdoc_file, write_mdoc_file
from .ts_filter import filter_names

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# blendmont intensity-correction options
# ---------------------------------------------------------------------------

def build_intensity_args(fix_from_edges=0, base=None, sum_for_gradient=False,
                         other_gradient_file=None, flatfield_file=None):
    """Return the blendmont intensity-correction flags as a list of strings.

    Maps directly onto blendmont's INTENSITY CORRECTION OPTIONS:

    ==================  =========================================
    Parameter           blendmont flag
    ==================  =========================================
    fix_from_edges      -intensity {1,2}   (0 emits nothing)
    base                -base {value}
    sum_for_gradient    -sum
    other_gradient_file -other {path}
    flatfield_file      -flatfield {path}
    ==================  =========================================

    Returns ``[]`` when nothing is enabled, so the generated command is
    byte-for-byte identical to the pre-existing one.

    Flags are emitted in a deterministic order (-intensity, -base, -sum,
    -other, -flatfield) so tests and the logged .sh files are stable. File
    paths are quoted with :func:`shlex.quote` because the command is run
    through ``shell=True``.

    Raises
    ------
    ValueError
        If ``fix_from_edges`` is not one of {0, 1, 2}, or if both
        ``sum_for_gradient`` and ``other_gradient_file`` are set (blendmont's
        -sum and -other cannot be combined).
    """
    if fix_from_edges not in (0, 1, 2):
        raise ValueError(
            f"fix_from_edges must be 0, 1, or 2 (got {fix_from_edges!r})"
        )
    if sum_for_gradient and other_gradient_file:
        raise ValueError(
            "sum_for_gradient and other_gradient_file are mutually exclusive: "
            "blendmont's -sum and -other cannot be combined"
        )
    if base is not None and fix_from_edges == 0:
        logger.warning(
            "intensity base=%s is set but fix_from_edges is 0; -base has no "
            "effect without -intensity", base
        )

    args = []
    if fix_from_edges:
        args += ["-intensity", str(fix_from_edges)]
    if base is not None:
        args += ["-base", f"{float(base):g}"]
    if sum_for_gradient:
        args += ["-sum"]
    if other_gradient_file:
        args += ["-other", shlex.quote(str(other_gradient_file))]
    if flatfield_file:
        args += ["-flatfield", shlex.quote(str(flatfield_file))]
    return args


def intensity_args_from_config(cfg_block, data_dir):
    """Build blendmont intensity flags from a YAML ``blend.intensity`` block.

    ``cfg_block`` is the ``intensity`` mapping (or None/{} when absent).
    ``other_gradient_file`` and ``flatfield_file`` are resolved against
    ``data_dir`` (unless already absolute) and must exist on disk.

    This is where up-front validation happens — call it once before any
    tilt-series is processed, not per tilt.

    Raises
    ------
    FileNotFoundError
        If a resolved ``other_gradient_file`` / ``flatfield_file`` does not
        exist (the message names the config key).
    ValueError
        Propagated from :func:`build_intensity_args`.
    """
    cfg_block = cfg_block or {}
    # Default is 1 (solve per-piece scaling from the overlap zones) when the
    # key is absent or null. An explicit 0 still disables it — don't collapse
    # 0 into the default the way ``x or 1`` would.
    raw_fix             = cfg_block.get("fix_from_edges", 1)
    fix_from_edges      = 1 if raw_fix is None else int(raw_fix)
    base                = cfg_block.get("base", None)
    sum_for_gradient    = bool(cfg_block.get("sum_for_gradient", False))
    other_gradient_file = cfg_block.get("other_gradient_file", None)
    flatfield_file      = cfg_block.get("flatfield_file", None)

    def _resolve_existing(rel, key):
        if not rel:
            return None
        path = rel if os.path.isabs(rel) else os.path.join(data_dir, rel)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"blend.intensity.{key} not found: {path}"
            )
        return path

    other_resolved     = _resolve_existing(other_gradient_file, "other_gradient_file")
    flatfield_resolved = _resolve_existing(flatfield_file,       "flatfield_file")

    return build_intensity_args(
        fix_from_edges=fix_from_edges,
        base=base,
        sum_for_gradient=sum_for_gradient,
        other_gradient_file=other_resolved,
        flatfield_file=flatfield_resolved,
    )


def _normalize_tiles_to_center(image_list, shifts_list, processing_dir, prefix,
                               frame_idx=None):
    """Linearly rescale off-center tile images to match the center tile's histogram.

    The center tile is identified as the entry whose pixel shift is closest to
    [0, 0].  Each off-center tile is rescaled so that its mean and standard
    deviation match those of the center tile.

    Parameters
    ----------
    image_list : list[str]
        Paths to the tile MRC files.  For averages these are 2-D single-slice
        MRCs; for frame stacks they are 3-D MRCs.
    shifts_list : list[[int, int, int]]
        Pixel shifts for each tile, same order as *image_list*.
    processing_dir : str
        Directory where normalised temp MRC files are written.
    prefix : str
        Filename stem used when creating temp files.
    frame_idx : int or None
        When not None, extract and normalise only the z-slice at this index
        from each 3-D MRC, writing single-slice temp files.  The caller must
        then pass ``frame_num=0`` to :func:`imod_newstack`.
        When None, treat each MRC as a 2-D image (average blending).

    Returns
    -------
    list[str]
        New image list.  Off-center tiles are replaced by normalised temp
        file paths.  When *frame_idx* is given, **all** tiles are replaced by
        single-slice temp files so the caller can uniformly use ``frame_num=0``.
    """
    import mrcfile
    import numpy as np

    # Identify centre tile: minimum L1 norm of (row, col) pixel shift
    dists = [abs(s[0]) + abs(s[1]) for s in shifts_list]
    center_idx = int(np.argmin(dists))

    # Read centre tile data and compute reference statistics
    try:
        with mrcfile.open(image_list[center_idx], mode='r', permissive=True) as mrc:
            raw_center = mrc.data.astype(np.float32)
        ref_slice = raw_center[frame_idx] if frame_idx is not None else raw_center
        ref_mean  = float(np.mean(ref_slice))
        ref_std   = float(np.std(ref_slice))
    except Exception as exc:
        print(f"  [WARNING] normalize_to_center: cannot read centre tile "
              f"{image_list[center_idx]}: {exc}")
        return image_list  # fall back to originals

    new_image_list = list(image_list)

    # When frame_idx is given we must write a single-slice file for every tile
    # (including the centre) so the caller can pass frame_num=0 uniformly.
    indices_to_write = (
        range(len(image_list))
        if frame_idx is not None
        else [i for i in range(len(image_list)) if i != center_idx]
    )

    for i in indices_to_write:
        out_path = os.path.join(processing_dir, f"{prefix}_tile{i}.mrc")
        try:
            with mrcfile.open(image_list[i], mode='r', permissive=True) as mrc:
                raw_i      = mrc.data.astype(np.float32)
                voxel_sz_i = mrc.voxel_size
            src_slice = raw_i[frame_idx] if frame_idx is not None else raw_i

            if i == center_idx:
                # Centre tile: write the slice unchanged (needed for frame mode
                # so all outputs are single-slice MRCs)
                normed = src_slice
            else:
                src_mean = float(np.mean(src_slice))
                src_std  = float(np.std(src_slice))
                if src_std > 0:
                    normed = (src_slice - src_mean) / src_std * ref_std + ref_mean
                else:
                    # Flat image — shift to reference mean only
                    normed = src_slice - src_mean + ref_mean

            with mrcfile.new(out_path, overwrite=True) as mrc_out:
                mrc_out.set_data(normed)
                mrc_out.voxel_size = voxel_sz_i

            new_image_list[i] = out_path

        except Exception as exc:
            print(f"  [WARNING] normalize_to_center: failed for tile {i} "
                  f"({image_list[i]}): {exc}")
            # Keep original path as fallback

    return new_image_list


def _snap_shifts_to_uniform(shifts_list):
    """Snap each ``[x, y, z]`` shift to the nearest integer multiple of the
    smallest non-zero step per axis.

    SerialEM mdocs sometimes report ``PixelShiftFromCenter`` values that
    are 1–2 px off from a uniform grid (e.g. ``3682`` and ``7365`` instead
    of ``3682`` and ``7364``). IMOD's ``blendmont`` requires uniformly-
    spaced shifts, so each shift is snapped to ``round(s / step) * step``
    per axis. The step is taken as the smallest non-zero absolute shift in
    that axis, which corresponds to the 1-step-from-center tiles in a
    regular montage. Axes with all-zero shifts (1-D montage) are passed
    through unchanged.

    Returns a new list — the input is not modified.
    """
    if not shifts_list:
        return shifts_list

    abs_x = [abs(s[0]) for s in shifts_list if s[0]]
    abs_y = [abs(s[1]) for s in shifts_list if s[1]]
    step_x = min(abs_x) if abs_x else 0
    step_y = min(abs_y) if abs_y else 0

    snapped = []
    for s in shifts_list:
        nx = int(round(s[0] / step_x)) * step_x if step_x else int(s[0])
        ny = int(round(s[1] / step_y)) * step_y if step_y else int(s[1])
        z  = s[2] if len(s) > 2 else 0
        snapped.append([nx, ny, z])
    return snapped


def _write_command_log(log_path, cmd, result, note=None):
    """Write one IMOD command's cmd string, return code, stdout, and stderr
    to ``log_path``.

    ``note`` is an optional free-text line (e.g. the edge-function cleanup
    message) recorded above the return code, so per-command housekeeping is
    captured in the log instead of cluttering stdout.

    Called once per IMOD invocation when a log directory is configured.
    Silently no-ops if ``log_path`` is None so callers don't need to guard.
    """
    if log_path is None:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    stdout = result.stdout.decode(errors='replace') if result.stdout else ''
    stderr = result.stderr.decode(errors='replace') if result.stderr else ''
    with open(log_path, 'w') as f:
        f.write(f"# Command:\n{cmd}\n\n")
        if note:
            f.write(f"# Note: {note}\n\n")
        f.write(f"# Return code: {result.returncode}\n\n")
        f.write("# ── stdout ──────────────────────────────────────────────────\n")
        f.write(stdout)
        if stdout and not stdout.endswith('\n'):
            f.write('\n')
        f.write("\n# ── stderr ──────────────────────────────────────────────────\n")
        f.write(stderr)
        if stderr and not stderr.endswith('\n'):
            f.write('\n')


def write_plin(shifts_list, output_file):
    """Write pixel-shift list to a .plin file for blendmont."""
    with open(output_file, 'w') as f:
        for shift in shifts_list:
            f.write(f"{shift[0]} {shift[1]} 0\n")


def write_fileinlist(image_list, frame_num, output_file):
    """Write an IMOD newstack filein list (alternating path / z-index)."""
    with open(output_file, 'w') as f:
        f.write(f"{len(image_list)}\n")
        for image in image_list:
            f.write(f"{image}\n")
            f.write(f"{frame_num}\n")


def imod_newstack(image_list, frame_num, stack_out, processing_dir, log_path=None):
    """Stack images with IMOD newstack.

    Parameters
    ----------
    log_path : str or None
        If given, write the executed command, return code, stdout, and stderr
        to this file.

    Returns
    -------
    tuple of (CompletedProcess, str)
        The subprocess result and the exact shell command that was run, so
        the caller can record it for the per-tilt-series sh_files log.
    """
    stem = Path(stack_out).stem
    filein = os.path.join(processing_dir, f"{stem}_{frame_num}.filein")
    write_fileinlist(image_list, frame_num, filein)
    cmd = f"newstack -filein {filein} -output {stack_out}"
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
    )
    _write_command_log(log_path, cmd, result)
    if result.returncode != 0:
        print(f"  [WARNING] newstack failed for {stack_out}:\n"
              f"  {result.stderr.decode().strip()}")
    return result, cmd


def imod_blendmont(stk_file, plin_file, plout_file, blend_size,
                   blended_output, processing_dir,
                   blend_log_path=None, clip_log_path=None, *,
                   intensity_args=()):
    """Blend a montage stack and resize with IMOD blendmont + clip.

    Parameters
    ----------
    blend_log_path : str or None
        If given, write the blendmont command, return code, stdout, and stderr
        to this file.
    clip_log_path : str or None
        Same as ``blend_log_path`` but for the clip resize step.
    intensity_args : sequence of str
        Extra blendmont intensity-correction flags (see
        :func:`build_intensity_args`) appended after ``-shift``. Empty (the
        default) reproduces the original command exactly. When non-empty, any
        cached edge-function files (``*.ecd``/``*.xef``/``*.yef``) for this
        rootname are deleted first, because changing gradient options
        invalidates them and blendmont would otherwise silently reuse stale
        ones. The cleanup is recorded in the blendmont command log (not printed
        to stdout).

    Returns
    -------
    tuple of (CompletedProcess, CompletedProcess, list[str])
        The blendmont result, the clip result, and the two shell command
        strings (blendmont then clip) for the caller to record in the
        per-tilt-series sh_files log.
    """
    rootname = Path(blended_output).stem
    intermediate = os.path.join(processing_dir, f"{rootname}_raw.mrc")

    # Changing any intensity/gradient option invalidates cached edge
    # functions. blendmont writes them to the -roo root and silently reuses
    # them if present, so remove any stale ones first. The cleanup is recorded
    # in the blendmont command log (via ``note`` below) rather than printed, so
    # it doesn't clutter stdout / the progress bar on every tilt.
    edge_note = None
    if intensity_args:
        stale = []
        for ext in ('ecd', 'xef', 'yef'):
            stale.extend(glob.glob(os.path.join(processing_dir, f"{rootname}*.{ext}")))
        for f in stale:
            try:
                os.remove(f)
            except OSError:
                pass
        if stale:
            edge_note = (f"intensity options set — removed {len(stale)} cached "
                         f"edge-function file(s) for {rootname} so blendmont "
                         f"recomputes them")

    blend_cmd = (
        f"blendmont -imin {stk_file} -plin {plin_file} "
        f"-imout {intermediate} "
        f"-roo {os.path.join(processing_dir, rootname)} "
        f"-al {plout_file} -adj -shift"
    )
    if intensity_args:
        blend_cmd += " " + " ".join(intensity_args)
    result_blend = subprocess.run(
        blend_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
    )
    _write_command_log(blend_log_path, blend_cmd, result_blend, note=edge_note)
    if result_blend.returncode != 0:
        print(f"  [WARNING] blendmont failed:\n  {result_blend.stderr.decode().strip()}")

    clip_cmd = f"clip resize -ox {blend_size} -oy {blend_size} {intermediate} {blended_output}"
    result_clip = subprocess.run(
        clip_cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
    )
    _write_command_log(clip_log_path, clip_cmd, result_clip)
    if result_clip.returncode != 0:
        print(f"  [WARNING] clip failed:\n  {result_clip.stderr.decode().strip()}")

    return result_blend, result_clip, [blend_cmd, clip_cmd]


def discover_tilt_series(mdoc_dir):
    """Return sorted list of unique tilt-series base names from mdoc_dir."""
    corner_mdocs = glob.glob(os.path.join(mdoc_dir, "*ts*_0_0*.mrc.mdoc"))
    if not corner_mdocs:
        return []
    ts_set = {os.path.basename(f).split('_0_0')[0] for f in corner_mdocs}
    return sorted(ts_set)


# ---------------------------------------------------------------------------
# Per-tilt parallel worker — must be at module level to be picklable
# ---------------------------------------------------------------------------

def _blend_tilt_worker(args):
    """Blend one tilt angle (newstack → blendmont → clip) for all tiles.

    ``args`` is a tuple:
    (tilt_i, tile_sections,
     ts, cropped_averages_abs, cropped_frames_abs,
     processing_averages_dir, processing_frames_dir,
     output_averages_dir, output_frames_dir,
     blend_size, blend_frames, num_frames,
     normalize_averages_to_center, normalize_frames_to_center,
     log_dir, snap_shifts_to_grid, intensity_args)

    ``intensity_args`` is a tuple/list of blendmont intensity-correction
    flags (see :func:`build_intensity_args`), appended to both the averages
    and per-frame blendmont commands. Empty () reproduces the original
    command. It must stay the *last* element of the tuple — the tuple is
    pickled for the multiprocessing path.

    ``log_dir`` is the directory where one log file per IMOD command is
    written, named ``{ts}_{tilt}_{command}.log``. Pass None to disable.

    ``snap_shifts_to_grid`` (bool) controls whether the per-tile pixel
    shifts read from the mdocs are snapped to a uniform grid before being
    written to the .plin file. blendmont requires uniform spacing, but
    SerialEM mdocs can be 1–2 px off; default True.

    Returns
    -------
    tuple of (tilt_i, tilt_angle, blended_avg_path,
              blended_frames_path_or_None, commands)
        ``commands`` is a list of the shell command strings that were issued
        for this tilt, in the order they ran. The caller writes them to the
        per-tilt-series sh_files log.
    """
    (tilt_i, tile_sections,
     ts, cropped_averages_abs, cropped_frames_abs,
     processing_averages_dir, processing_frames_dir,
     output_averages_dir, output_frames_dir,
     blend_size, blend_frames, num_frames,
     normalize_averages_to_center, normalize_frames_to_center,
     log_dir, snap_shifts_to_grid, intensity_args) = args

    shifts_list = []
    image_list  = []
    tilt_angle  = tilt_i   # fallback
    commands    = []       # IMOD commands issued for this tilt, in order

    for section in tile_sections:
        tilt_angle = section.get('TiltAngle', tilt_i)
        shifts     = section.get('PixelShiftFromCenter', [0, 0])
        shifts_list.append([int(shifts[0]), int(shifts[1]), 0])
        subframe   = PureWindowsPath(section.get('SubFramePath', ''))
        image_list.append(
            os.path.join(cropped_averages_abs, subframe.name.replace('.tif', '.mrc'))
        )

    # SerialEM occasionally writes PixelShiftFromCenter values that are
    # 1–2 px off the regular grid (e.g. 3682 and 7365 instead of 3682 and
    # 7364). blendmont rejects non-uniform spacings, so snap each tile's
    # shift to the nearest multiple of the per-axis step.
    if snap_shifts_to_grid:
        shifts_list = _snap_shifts_to_uniform(shifts_list)

    # Helper closure: produce {ts}_{tilt}_{command}.log path under log_dir,
    # or None when logging is disabled.
    def _log(name):
        if log_dir is None:
            return None
        return os.path.join(log_dir, f"{ts}_{tilt_angle}_{name}.log")

    # Optionally normalise off-center averages to match the centre tile histogram
    if normalize_averages_to_center:
        image_list = _normalize_tiles_to_center(
            image_list, shifts_list,
            processing_averages_dir,
            f"{ts}_{tilt_angle}_norm",
        )

    plin_file   = os.path.join(processing_averages_dir, f"{ts}_{tilt_angle}.plin")
    plout_file  = os.path.join(processing_averages_dir, f"{ts}_{tilt_angle}.plout")
    stack_file  = os.path.join(processing_averages_dir, f"{ts}_{tilt_angle}.mrc")
    blended_out = os.path.join(output_averages_dir,     f"{ts}_{tilt_angle}_blended.mrc")

    write_plin(shifts_list, plin_file)
    _, ns_cmd = imod_newstack(image_list, 0, stack_file, processing_averages_dir,
                              log_path=_log('newstack'))
    commands.append(ns_cmd)
    _, _, bm_cmds = imod_blendmont(stack_file, plin_file, plout_file, blend_size,
                                   blended_out, processing_averages_dir,
                                   blend_log_path=_log('blendmont'),
                                   clip_log_path=_log('clip'),
                                   intensity_args=intensity_args)
    commands.extend(bm_cmds)

    frame_stack_out = None
    if blend_frames:
        frame_output_list = []
        for frame_i in range(num_frames):
            # Per-frame shifts are identical to the averages shifts (already
            # snapped above), so we only need to (re)build the frame file
            # paths here.
            frame_image_list = []
            for section in tile_sections:
                subframe = PureWindowsPath(section.get('SubFramePath', ''))
                frame_image_list.append(
                    os.path.join(cropped_frames_abs, subframe.name.replace('.tif', '.mrc'))
                )

            frame_stack   = os.path.join(processing_frames_dir, f"{ts}_{tilt_angle}_frame{frame_i}.mrc")
            frame_plin    = os.path.join(processing_frames_dir, f"{ts}_{tilt_angle}_frame{frame_i}.plin")
            frame_plout   = os.path.join(processing_frames_dir, f"{ts}_{tilt_angle}_frame{frame_i}.plout")
            frame_blended = os.path.join(processing_frames_dir, f"{ts}_{tilt_angle}_frame{frame_i}_blended.mrc")

            # Optionally normalise each frame: extract frame_i slice, match to
            # centre tile's same slice, write single-slice temp files
            if normalize_frames_to_center:
                frame_image_list = _normalize_tiles_to_center(
                    frame_image_list, shifts_list,
                    processing_frames_dir,
                    f"{ts}_{tilt_angle}_frame{frame_i}_norm",
                    frame_idx=frame_i,
                )
                frame_num_for_stack = 0   # temp files are single-slice
            else:
                frame_num_for_stack = frame_i

            write_plin(shifts_list, frame_plin)
            _, ns_cmd = imod_newstack(frame_image_list, frame_num_for_stack,
                                      frame_stack, processing_frames_dir,
                                      log_path=_log(f'frame{frame_i}_newstack'))
            commands.append(ns_cmd)
            _, _, bm_cmds = imod_blendmont(frame_stack, frame_plin, frame_plout, blend_size,
                                           frame_blended, processing_frames_dir,
                                           blend_log_path=_log(f'frame{frame_i}_blendmont'),
                                           clip_log_path=_log(f'frame{frame_i}_clip'),
                                           intensity_args=intensity_args)
            commands.extend(bm_cmds)
            frame_output_list.append(os.path.abspath(frame_blended))

        frame_stack_out = os.path.join(output_frames_dir, f"{ts}_{tilt_angle}_blended_frames.mrc")
        _, ns_cmd = imod_newstack(frame_output_list, 0, frame_stack_out, processing_frames_dir,
                                  log_path=_log('frames_newstack'))
        commands.append(ns_cmd)

    return (tilt_i, tilt_angle,
            os.path.abspath(blended_out),
            os.path.abspath(frame_stack_out) if frame_stack_out else None,
            commands)


# ---------------------------------------------------------------------------
# Main tilt-series processing function
# ---------------------------------------------------------------------------

def process_tilt_series(ts, mdoc_dir, cropped_averages_dir, cropped_frames_dir,
                        processing_averages_dir, processing_frames_dir,
                        output_averages_dir, output_frames_dir,
                        output_averages_mdoc_dir, output_frames_mdoc_dir,
                        blend_size, blend_frames, num_frames,
                        normalize_averages_to_center=False,
                        normalize_frames_to_center=False,
                        num_workers=1, show_progress=True, tqdm_position=0,
                        sh_files_dir=None, log_dir=None,
                        snap_shifts_to_grid=True, intensity_args=()):
    """Blend all tiles for one tilt-series.

    Parameters
    ----------
    normalize_averages_to_center : bool
        When True, linearly rescale each off-center average tile image so that
        its mean and standard deviation match those of the center tile (_0_0)
        before blending.
    normalize_frames_to_center : bool
        When True, apply the same per-frame histogram normalisation to the
        per-frame stacks (only relevant when blend_frames is True).
    num_workers : int
        Number of parallel workers for per-tilt blending.
        1 = sequential (default). >1 = parallel via ProcessPoolExecutor.
    show_progress : bool
        Show a per-tilt tqdm progress bar.
    tqdm_position : int
        tqdm ``position`` for the inner bar (use 1 when an outer bar is at 0).
    sh_files_dir : str or None
        Directory where a per-tilt-series ``{ts}.sh`` script is written
        containing every IMOD command (newstack / blendmont / clip) issued
        for that tilt-series, ordered by tilt index. When None (default),
        a sibling ``sh_files/`` of ``processing_averages_dir`` is used so
        the file ends up at ``processing/sh_files/{ts}.sh``.
    log_dir : str or None
        Directory where one log file per IMOD command is written, named
        ``{ts}_{tilt}_{command}.log``. Each log contains the executed
        command, return code, and full stdout + stderr. When None (default),
        a sibling ``log/`` of ``processing_averages_dir`` is used so files
        end up at ``processing/log/``.
    snap_shifts_to_grid : bool
        When True (default), snap each tile's ``PixelShiftFromCenter`` to
        the nearest integer multiple of the per-axis step before writing
        the .plin file. SerialEM can write shifts that are 1–2 px off the
        regular grid (e.g. 3682 and 7365 instead of 3682 and 7364), and
        blendmont refuses to run on non-uniform spacings.
    intensity_args : sequence of str
        blendmont intensity-correction flags (see
        :func:`build_intensity_args`) applied identically to the averages and
        per-frame blendmont commands. Empty (default) leaves the command
        unchanged. These are independent of and composable with the
        ``normalize_*_to_center`` options above.
    """
    if sh_files_dir is None:
        sh_files_dir = os.path.join(
            os.path.dirname(os.path.abspath(processing_averages_dir)),
            'sh_files',
        )
    os.makedirs(sh_files_dir, exist_ok=True)

    if log_dir is None:
        log_dir = os.path.join(
            os.path.dirname(os.path.abspath(processing_averages_dir)),
            'log',
        )
    os.makedirs(log_dir, exist_ok=True)

    tile_mdoc_paths = sorted(glob.glob(os.path.join(mdoc_dir, f"{ts}_*_*.mrc.mdoc")))
    if not tile_mdoc_paths:
        print(f"  [WARNING] No tile MDOCs found for {ts}, skipping.")
        return

    tile_mdocs = []
    output_mdoc = None
    output_frame_mdoc = None

    for tile_path in tile_mdoc_paths:
        parsed = parse_mdoc_file(tile_path)
        tile_mdocs.append(parsed)
        if "_0_0" in os.path.basename(tile_path):
            output_mdoc = parse_mdoc_file(tile_path)
            if blend_frames:
                output_frame_mdoc = parse_mdoc_file(tile_path)

    if output_mdoc is None:
        print(f"  [WARNING] Corner tile (_0_0) MDOC not found for {ts}, skipping.")
        return

    num_tilts = len(tile_mdocs[0]['z_sections'])
    num_tiles = len(tile_mdocs)
    print(f"  {ts}: {num_tiles} tiles × {num_tilts} tilts  (workers={num_workers})")

    cropped_averages_abs = os.path.abspath(cropped_averages_dir)
    cropped_frames_abs   = os.path.abspath(cropped_frames_dir)

    # Build one args-tuple per tilt angle
    task_args = [
        (tilt_i,
         [tm['z_sections'][tilt_i] for tm in tile_mdocs],
         ts, cropped_averages_abs, cropped_frames_abs,
         processing_averages_dir, processing_frames_dir,
         output_averages_dir, output_frames_dir,
         blend_size, blend_frames, num_frames,
         normalize_averages_to_center, normalize_frames_to_center,
         log_dir, snap_shifts_to_grid, tuple(intensity_args))
        for tilt_i in range(num_tilts)
    ]

    if show_progress:
        pbar = tqdm.tqdm(total=num_tilts,
                         desc=f"  {ts}",
                         position=tqdm_position,
                         leave=True)
    else:
        pbar = None

    results = {}   # tilt_i → (tilt_angle, blended_avg, blended_frames, commands)

    if num_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_blend_tilt_worker, a): a[0] for a in task_args}
            for future in concurrent.futures.as_completed(futures):
                tilt_i, tilt_angle, avg_path, frm_path, commands = future.result()
                results[tilt_i] = (tilt_angle, avg_path, frm_path, commands)
                if pbar is not None:
                    pbar.set_postfix(angle=f"{tilt_angle:+.1f}°")
                    pbar.update(1)
    else:
        for args in task_args:
            tilt_i, tilt_angle, avg_path, frm_path, commands = _blend_tilt_worker(args)
            results[tilt_i] = (tilt_angle, avg_path, frm_path, commands)
            if pbar is not None:
                pbar.set_postfix(angle=f"{tilt_angle:+.1f}°")
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    # Write a re-runnable shell script of every IMOD command issued for
    # this tilt-series, ordered by tilt index. Useful both for reproducing
    # the run by hand and for debugging when an IMOD step fails.
    sh_path = os.path.join(sh_files_dir, f"{ts}.sh")
    with open(sh_path, 'w') as f:
        f.write("#!/usr/bin/env bash\n")
        f.write(f"# IMOD commands for tilt-series: {ts}\n")
        f.write("# Generated by sqap-montage blend step.\n")
        f.write("# Re-run from the same working directory used for the original blend.\n")
        f.write("# The .plin and .filein files these commands reference are written\n")
        f.write("# by sqap-montage and must exist (or be re-created) before re-running.\n")
        f.write("set -e\n\n")
        for tilt_i in sorted(results.keys()):
            tilt_angle, _, _, commands = results[tilt_i]
            f.write(f"# ── Tilt {tilt_i}: angle={tilt_angle} ──\n")
            for cmd in commands:
                f.write(f"{cmd}\n")
            f.write("\n")
    try:
        os.chmod(sh_path, 0o755)
    except OSError:
        pass

    # Update mdoc with output paths (must happen after all workers finish)
    for tilt_i, (tilt_angle, avg_path, frm_path, _commands) in results.items():
        output_mdoc['z_sections'][tilt_i]['SubFramePath'] = avg_path
        if blend_frames and output_frame_mdoc is not None and frm_path:
            output_frame_mdoc['z_sections'][tilt_i]['SubFramePath'] = frm_path

    write_mdoc_file(output_mdoc,
                    os.path.join(output_averages_mdoc_dir, f"{ts}_blended.mrc.mdoc"))
    if blend_frames and output_frame_mdoc is not None:
        write_mdoc_file(output_frame_mdoc,
                        os.path.join(output_frames_mdoc_dir, f"{ts}_blended_frames.mrc.mdoc"))


@click.command()
@click.option('--mdoc-dir',      default='mdocs',            show_default=True,
              help='Directory containing per-tile .mrc.mdoc files.')
@click.option('--averages-dir',  default='cropped/averages', show_default=True,
              help='Directory of cropped motion-corrected average MRCs.')
@click.option('--frames-dir',    default='cropped/frames',   show_default=True,
              help='Directory of cropped per-exposure frame MRC stacks.')
@click.option('--output-dir',    default='blended',          show_default=True,
              help='Root output directory.')
@click.option('--processing-dir',default='processing',       show_default=True,
              help='Directory for intermediate files.')
@click.option('--blend-size',    default=11664, show_default=True,
              help='Output image edge length in pixels after clip resize.')
@click.option('--blend-frames/--no-blend-frames', default=True, show_default=True,
              help='Also blend per-frame stacks.')
@click.option('--num-frames',    default=4, show_default=True,
              help='Number of frames per exposure (used with --blend-frames).')
@click.option('--normalize-averages-to-center/--no-normalize-averages-to-center',
              default=False, show_default=True,
              help=('Linearly rescale each off-center average tile so its mean and '
                    'standard deviation match the center tile before blending.'))
@click.option('--normalize-frames-to-center/--no-normalize-frames-to-center',
              default=False, show_default=True,
              help=('Linearly rescale each off-center frame tile so its mean and '
                    'standard deviation match the center tile before blending. '
                    'Only applies when --blend-frames is set.'))
@click.option('--ts', 'ts_filter', default=None, multiple=True,
              help=('Process only tilt-series matching these names/globs '
                    '(repeatable, wildcards * ? [..] allowed). Defaults to all.'))
@click.option('--log-dir', default=None, show_default=True,
              help=('Directory for per-IMOD-command log files '
                    '({ts}_{tilt}_{command}.log). Defaults to '
                    '<processing-dir>/log.'))
@click.option('--sh-files-dir', default=None, show_default=True,
              help=('Directory for per-tilt-series shell scripts '
                    '({ts}.sh). Defaults to <processing-dir>/sh_files.'))
@click.option('--snap-shifts-to-grid/--no-snap-shifts-to-grid',
              default=True, show_default=True,
              help=('Snap PixelShiftFromCenter values to a uniform grid '
                    'before writing the .plin file. blendmont requires '
                    'uniform spacing; SerialEM mdocs can be 1–2 px off.'))
@click.option('--intensity', '--fix-intensity-from-edges', 'fix_from_edges',
              type=int, default=1, show_default=True,
              help=('blendmont -intensity: 0=off, 1=solve scaling from '
                    'overlap-zone differences (default), 2=also fit/remove a '
                    'planar gradient first.'))
@click.option('--intensity-base', 'intensity_base',
              type=float, default=None,
              help=('blendmont -base: value subtracted before scaling and '
                    'added back after. Only meaningful with --intensity > 0.'))
@click.option('--sum-for-gradient/--no-sum-for-gradient',
              default=False, show_default=True,
              help=('blendmont -sum: sum all pieces, run "clip planefit", and '
                    'correct every piece for the planar gradient. Mutually '
                    'exclusive with --other-gradient-file.'))
@click.option('--other-gradient-file', 'other_gradient_file',
              type=click.Path(), default=None,
              help='blendmont -other: pre-computed planar gradient file.')
@click.option('--flatfield-file', 'flatfield_file',
              type=click.Path(), default=None,
              help='blendmont -flatfield: flatfield image to scale by.')
def main(mdoc_dir, averages_dir, frames_dir, output_dir, processing_dir,
         blend_size, blend_frames, num_frames,
         normalize_averages_to_center, normalize_frames_to_center, ts_filter,
         log_dir, sh_files_dir, snap_shifts_to_grid,
         fix_from_edges, intensity_base, sum_for_gradient,
         other_gradient_file, flatfield_file):
    """Blend 3×3 montage tile images into a single giant tilt-series."""
    intensity_args = build_intensity_args(
        fix_from_edges=fix_from_edges,
        base=intensity_base,
        sum_for_gradient=sum_for_gradient,
        other_gradient_file=other_gradient_file,
        flatfield_file=flatfield_file,
    )
    out_avg      = os.path.join(output_dir, 'averages')
    out_frm      = os.path.join(output_dir, 'frames')
    out_avg_mdoc = os.path.join(out_avg, 'mdocs')
    out_frm_mdoc = os.path.join(out_frm, 'mdocs')
    proc_avg     = os.path.join(processing_dir, 'blending_averages')
    proc_frm     = os.path.join(processing_dir, 'blending_frames')

    for d in [out_avg, out_avg_mdoc, proc_avg]:
        os.makedirs(d, exist_ok=True)
    if blend_frames:
        for d in [out_frm, out_frm_mdoc, proc_frm]:
            os.makedirs(d, exist_ok=True)

    ts_list = discover_tilt_series(mdoc_dir)
    if not ts_list:
        print(f"No tilt-series found in '{mdoc_dir}'. Exiting.")
        sys.exit(1)

    if ts_filter:
        ts_list = filter_names(ts_list, ts_filter)
        if not ts_list:
            print("No tilt-series matched the --ts filter. Exiting.")
            sys.exit(1)

    print(f"Found {len(ts_list)} tilt-series to process.")

    for i, ts in enumerate(ts_list):
        print(f"\n[{i + 1}/{len(ts_list)}] {ts}")
        process_tilt_series(
            ts=ts,
            mdoc_dir=mdoc_dir,
            cropped_averages_dir=averages_dir,
            cropped_frames_dir=frames_dir,
            processing_averages_dir=proc_avg,
            processing_frames_dir=proc_frm,
            output_averages_dir=out_avg,
            output_frames_dir=out_frm,
            output_averages_mdoc_dir=out_avg_mdoc,
            output_frames_mdoc_dir=out_frm_mdoc,
            blend_size=blend_size,
            blend_frames=blend_frames,
            num_frames=num_frames,
            normalize_averages_to_center=normalize_averages_to_center,
            normalize_frames_to_center=normalize_frames_to_center,
            sh_files_dir=sh_files_dir,
            log_dir=log_dir,
            snap_shifts_to_grid=snap_shifts_to_grid,
            intensity_args=intensity_args,
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
