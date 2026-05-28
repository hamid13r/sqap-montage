#!/usr/bin/env python3
"""
sqap_montage.py — Square Aperture Montage pipeline

Single entry-point script with subcommands for each processing step.
All parameters are read from a YAML config file.

Usage
-----
    python sqap_montage.py crop      --config pipeline.yaml
    python sqap_montage.py blend     --config pipeline.yaml
    python sqap_montage.py fill      --config pipeline.yaml
    python sqap_montage.py make-mdoc --config pipeline.yaml

Generate a template config:
    python sqap_montage.py write-config pipeline.yaml
"""

import concurrent.futures
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path, PureWindowsPath

import click
import yaml
from tqdm import tqdm

from square_aperture_montage.crop_images import crop_average, crop_frames_for_image
from square_aperture_montage.blend_tiles import discover_tilt_series, process_tilt_series
from square_aperture_montage.mdoc_reader import parse_mdoc_file, write_mdoc_file


# ─────────────────────────────────────────────────────────────────────────────
# Keys copied from the _0_0 reference tile into the blended mdoc
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_KEYS_TO_COPY = [
    'MinMaxMean', 'StagePosition', 'StageZ', 'Magnification', 'Intensity',
    'DoseRate', 'PixelSpacing', 'SpotSize', 'ProbeMode', 'ImageShift',
    'RotationAngle', 'TiltAngle', 'ExposureTime', 'Binning', 'CameraIndex',
    'DividedBy2', 'OperatingMode', 'UsingCDS', 'MagIndex', 'LowDoseConSet',
    'CountsPerElectron', 'TargetDefocus', 'NumSubFrames', 'FrameDosesAndNumber',
    'DateTime', 'FilterSlitAndLoss', 'UncroppedSize', 'RotationAndFlip',
    'TimeStamp', 'SpecimenShift', 'EucentricOffset', 'Ctfplotter',
]


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load and return the YAML config as a dict."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve(data_dir: str, rel_path: str) -> str:
    """Join data_dir with a relative path, or return rel_path if already absolute."""
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.join(data_dir, rel_path)


def _cleanup_dir(path: str, keep: bool) -> None:
    """Delete intermediate .mrc files inside a processing directory.

    Only removes *.mrc files — text files (.txt, .plin, .plout, .log, etc.)
    are left in place so the user can inspect them or delete the whole
    directory manually afterwards.

    Prints a summary line. Does nothing if the directory does not exist.
    """
    if not os.path.isdir(path):
        return
    if keep:
        click.echo(f"  [keep] intermediate files retained: {path}")
        return
    mrc_files = glob.glob(os.path.join(path, '**', '*.mrc'), recursive=True)
    for f in mrc_files:
        os.remove(f)
    click.echo(f"  [cleanup] removed {len(mrc_files)} intermediate .mrc file(s) from {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI group
# ─────────────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Square Aperture Montage — IMOD wrapper for 3×3 cryo-EM tilt-series stitching.

    \b
    Steps:
        crop      Remove blank aperture borders from each tile image
        blend     Stitch 3×3 tile images into one giant tilt-series
        fill      Fill blending-seam gaps with local texture (GPU)
        make-mdoc Build blended .mdoc files from per-tile mdoc files

    \b
    Example:
        python sqap_montage.py crop      --config pipeline.yaml
        python sqap_montage.py blend     --config pipeline.yaml
        python sqap_montage.py fill      --config pipeline.yaml
        python sqap_montage.py make-mdoc --config pipeline.yaml
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# crop
# ─────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True),
              help='Path to pipeline YAML config.')
def crop(config):
    """Crop blank aperture borders from each tile image.

    Reads the [crop] section of the config file.
    """
    cfg               = load_config(config)
    c                 = cfg.get('crop', {})
    data_dir          = cfg.get('data_dir', '.')
    keep_intermediate = cfg.get('keep_intermediate', False)
    num_workers       = int(cfg.get('num_workers', 1))

    input_dir       = resolve(data_dir, c.get('input_dir',      'frames/averages'))
    output_dir      = resolve(data_dir, c.get('output_dir',     'cropped'))
    processing_dir  = resolve(data_dir, c.get('processing_dir', 'processing/crop'))
    frames_dir      = resolve(data_dir, c.get('frames_dir',     'frames'))
    averages_suffix = c.get('averages_suffix', '')
    crop_frames     = c.get('crop_frames',    True)
    crop_x          = c.get('crop_x',         3840)
    crop_y          = c.get('crop_y',         3840)
    filter_window   = c.get('filter_window',  200)
    mask_threshold  = c.get('mask_threshold', 0.5)
    trim            = c.get('trim',           50)

    output_averages_dir = os.path.join(output_dir, 'averages')
    output_frames_dir   = os.path.join(output_dir, 'frames')

    if os.path.isdir(input_dir):
        image_files = sorted(glob.glob(os.path.join(input_dir, '*.mrc')))
    elif os.path.isfile(input_dir):
        image_files = [input_dir]
    else:
        click.echo(f"ERROR: '{input_dir}' is neither a file nor a directory.", err=True)
        sys.exit(1)

    if not image_files:
        click.echo(f"No .mrc files found in '{input_dir}'. Exiting.")
        return

    click.echo(f"Found {len(image_files)} image(s) to process.")

    if num_workers > 1:
        from square_aperture_montage.crop_images import _crop_image_worker
        click.echo(f"  Workers: {num_workers} (parallel)")
        task_args = [
            (image_file, output_averages_dir, processing_dir, output_frames_dir,
             frames_dir, crop_frames, averages_suffix,
             filter_window, mask_threshold, trim, crop_x, crop_y)
            for image_file in image_files
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_crop_image_worker, a) for a in task_args]
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(image_files), desc="Cropping"):
                try:
                    future.result()
                except Exception as exc:
                    click.echo(f"  [ERROR] {exc}", err=True)
    else:
        for image_file in tqdm(image_files, desc="Cropping"):
            x0, x1, y0, y1 = crop_average(
                image_file, output_averages_dir, processing_dir,
                filter_size=filter_window, mask_threshold=mask_threshold,
                trim=trim, crop_x=crop_x, crop_y=crop_y,
            )
            if crop_frames:
                crop_frames_for_image(image_file, frames_dir, output_frames_dir, x0, x1, y0, y1,
                                      averages_suffix=averages_suffix)

    click.echo("Crop done.")
    _cleanup_dir(processing_dir, keep_intermediate)


# ─────────────────────────────────────────────────────────────────────────────
# Preview stack helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_preview_stack(ts, averages_dir, preview_dir, binning, processing_dir):
    """Stack blended averages for one tilt-series sorted by tilt angle.

    Uses IMOD ``newstack -bin`` to assemble and bin in a single call.
    Output filename: ``{ts}_preview_bin{binning}.mrc``
    """
    files = glob.glob(os.path.join(averages_dir, f"{ts}_*_blended.mrc"))
    if not files:
        click.echo(f"  [WARNING] No blended averages found for {ts}, skipping preview.")
        return

    def _angle(f):
        # filename pattern: {ts}_{angle}_blended.mrc  e.g. ts_001_-52.0_blended.mrc
        try:
            return float(Path(f).stem.replace(f"{ts}_", "").replace("_blended", ""))
        except ValueError:
            return 0.0

    files_sorted = sorted(files, key=_angle)
    out = os.path.join(preview_dir, f"{ts}_preview_bin{binning}.mrc")
    filein = os.path.join(processing_dir, f"{ts}_preview.filein")

    with open(filein, "w") as fh:
        fh.write(f"{len(files_sorted)}\n")
        for img in files_sorted:
            fh.write(f"{img}\n0\n")

    result = subprocess.run(
        f"newstack -filein {filein} -bin {binning} -output {out}",
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        click.echo(f"  [WARNING] preview newstack failed for {ts}:\n"
                   f"  {result.stderr.decode().strip()}")
    else:
        click.echo(f"  ✓ {os.path.basename(out)}")


# ─────────────────────────────────────────────────────────────────────────────
# blend
# ─────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True),
              help='Path to pipeline YAML config.')
def blend(config):
    """Stitch 3×3 tile images into a single giant tilt-series.

    Reads the [blend] section of the config file.
    """
    cfg               = load_config(config)
    b                 = cfg.get('blend', {})
    data_dir          = cfg.get('data_dir', '.')
    keep_intermediate = cfg.get('keep_intermediate', False)
    num_workers       = int(cfg.get('num_workers', 1))

    mdoc_dir       = resolve(data_dir, b.get('mdoc_dir',       'mdocs'))
    averages_dir   = resolve(data_dir, b.get('averages_dir',   'cropped/averages'))
    frames_dir     = resolve(data_dir, b.get('frames_dir',     'cropped/frames'))
    output_dir     = resolve(data_dir, b.get('output_dir',     'blended'))
    processing_dir = resolve(data_dir, b.get('processing_dir', 'processing'))
    blend_size     = b.get('blend_size',   11664)
    blend_frames   = b.get('blend_frames', True)
    num_frames     = b.get('num_frames',   4)
    ts_filter      = b.get('ts_filter',    [])
    preview        = b.get('preview',         True)
    preview_bin    = int(b.get('preview_binning', 24))
    preview_dir    = resolve(data_dir, b.get('preview_dir', 'blended/previews'))
    # Optional user-set locations for the per-IMOD-command log files and the
    # per-tilt-series shell scripts. Either may be omitted from the config —
    # process_tilt_series then defaults them to <processing_dir>/log and
    # <processing_dir>/sh_files respectively.
    log_dir_cfg      = b.get('log_dir', None)
    sh_files_dir_cfg = b.get('sh_files_dir', None)
    log_dir      = resolve(data_dir, log_dir_cfg)      if log_dir_cfg      else None
    sh_files_dir = resolve(data_dir, sh_files_dir_cfg) if sh_files_dir_cfg else None
    # Snap PixelShiftFromCenter values to a uniform grid before writing
    # the .plin file. blendmont requires uniform spacing; SerialEM mdocs
    # can write values that are 1–2 px off. Default True.
    snap_shifts_to_grid = bool(b.get('snap_shifts_to_grid', True))

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
        click.echo(f"No tilt-series found in '{mdoc_dir}'. Exiting.")
        sys.exit(1)

    if ts_filter:
        ts_list = [ts for ts in ts_list if ts in ts_filter]
        if not ts_list:
            click.echo("No tilt-series matched ts_filter in config. Exiting.")
            sys.exit(1)

    click.echo(f"Found {len(ts_list)} tilt-series to process.")
    if num_workers > 1:
        click.echo(f"  Workers: {num_workers} per tilt-series (tilts blended in parallel)")

    with tqdm(total=len(ts_list), desc="Overall", position=0, leave=True) as overall:
        for ts in ts_list:
            overall.set_postfix(ts=ts)
            try:
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
                    num_workers=num_workers,
                    show_progress=True,
                    tqdm_position=1,
                    sh_files_dir=sh_files_dir,
                    log_dir=log_dir,
                    snap_shifts_to_grid=snap_shifts_to_grid,
                )
            except Exception as exc:
                click.echo(f"  [ERROR] {ts}: {exc}", err=True)
            overall.update(1)

    if preview:
        os.makedirs(preview_dir, exist_ok=True)
        click.echo("\nBuilding preview stacks…")
        for ts in tqdm(ts_list, desc="Previews"):
            _make_preview_stack(ts, out_avg, preview_dir, preview_bin, proc_avg)

    click.echo("\nBlend done.")
    # Only the two subdirs we created are cleaned up — not the whole processing/ parent
    _cleanup_dir(proc_avg, keep_intermediate)
    _cleanup_dir(proc_frm, keep_intermediate)


# ─────────────────────────────────────────────────────────────────────────────
# fill
# ─────────────────────────────────────────────────────────────────────────────

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True),
              help='Path to pipeline YAML config.')
def fill(config):
    """Fill blending-seam gaps with local texture (GPU-accelerated).

    Reads the [fill] section of the config file.
    """
    try:
        import torch
    except ImportError:
        click.echo(
            "ERROR: PyTorch is required for gap filling.\n"
            "Install with: pip install torch",
            err=True,
        )
        sys.exit(1)

    from square_aperture_montage.remove_gaps import (
        process_image, _process_with_gpu,
        DEFAULT_DETECT_KERNELS, DEFAULT_DILATE_KERNELS,
        _parse_kernels, _parse_index_list,
    )

    cfg      = load_config(config)
    f        = cfg.get('fill', {})
    data_dir = cfg.get('data_dir', '.')

    input_dir  = resolve(data_dir, f.get('input_dir',  'blended/frames'))
    output_dir = resolve(data_dir, f.get('output_dir', 'blended/frames_filled'))
    mask_dir   = resolve(data_dir, f.get('mask_dir',   'blended/frames_masks'))
    gpus       = str(f.get('gpus',     '0'))
    resume     = f.get('resume',    True)
    sigma      = float(f.get('sigma',  5.0))
    tile_num   = int(f.get('tile_num', 8))

    # Kernels
    detect_kernels = _parse_kernels(f.get('detect_kernels', DEFAULT_DETECT_KERNELS))
    dilate_kernels = _parse_kernels(f.get('dilate_kernels', DEFAULT_DILATE_KERNELS))

    # Manual seam positions — if set, auto-detection is skipped entirely.
    # Each entry may be an integer or an "start-end" range string.
    seam_rows = _parse_index_list(f.get('seam_rows', []))
    seam_cols = _parse_index_list(f.get('seam_cols', []))

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir,   exist_ok=True)

    image_list = sorted(glob.glob(os.path.join(input_dir, '*.mrc')))
    if not image_list:
        click.echo(f"No .mrc files found in '{input_dir}'. Exiting.")
        sys.exit(1)

    click.echo(f"Found {len(image_list)} images to process.")
    if seam_rows or seam_cols:
        click.echo(f"  Mode: manual seams — rows: {seam_rows}  cols: {seam_cols}")
    else:
        click.echo(f"  Mode: auto-detect — sigma: {sigma}")
        click.echo(f"  detect_kernels: {detect_kernels}")
    click.echo(f"  dilate_kernels: {dilate_kernels}")

    if gpus.strip().lower() == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
        for img in tqdm(image_list, desc="Filling gaps"):
            process_image(img, output_dir, mask_dir, device, resume, sigma, tile_num,
                          detect_kernels=detect_kernels, dilate_kernels=dilate_kernels,
                          seam_rows=seam_rows, seam_cols=seam_cols)
        click.echo("Fill done.")
        return

    available = torch.cuda.device_count()
    gpu_ids   = [int(g.strip()) for g in gpus.split(',') if g.strip().isdigit()]
    gpu_ids   = [g for g in gpu_ids if g < available] or [0]

    click.echo(f"Using GPUs: {gpu_ids}")
    task_args = [
        (img, gpu_ids[i % len(gpu_ids)], output_dir, mask_dir,
         resume, sigma, tile_num,
         detect_kernels, dilate_kernels,
         seam_rows, seam_cols)
        for i, img in enumerate(image_list)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(_process_with_gpu, a) for a in task_args]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(image_list),
            desc="Filling gaps",
        ):
            try:
                future.result()
            except Exception as exc:
                click.echo(f"  [ERROR] {exc}", err=True)

    click.echo("Fill done.")


# ─────────────────────────────────────────────────────────────────────────────
# make-mdoc
# ─────────────────────────────────────────────────────────────────────────────

@cli.command('make-mdoc')
@click.option('--config', required=True, type=click.Path(exists=True),
              help='Path to pipeline YAML config.')
def make_mdoc(config):
    """Build blended .mdoc files from the 9 per-tile mdoc files.

    Reads 9 tile mdocs from mdoc_dir (e.g. mdocs/), takes all metadata
    from the _0_0 corner tile, computes the accumulated ExposureDose, sets
    SubFramePath to the corresponding blended image, and writes one mdoc
    per tilt-series to output_dir (e.g. blended_mdocs/).

    Reads the [make_mdoc] section of the config file.
    """
    cfg      = load_config(config)
    m        = cfg.get('make_mdoc', {})
    data_dir = cfg.get('data_dir', '.')

    mdoc_dir      = resolve(data_dir, m.get('mdoc_dir',      'mdocs'))
    blended_dir   = resolve(data_dir, m.get('blended_dir',   'blended/averages'))
    output_dir    = resolve(data_dir, m.get('output_dir',    'blended_mdocs'))
    dose_per_tilt = float(m.get('dose_per_tilt', 3.0))
    keys_to_copy  = m.get('keys_to_copy', DEFAULT_KEYS_TO_COPY)
    ts_filter     = m.get('ts_filter', [])

    os.makedirs(output_dir, exist_ok=True)

    ts_list = discover_tilt_series(mdoc_dir)
    if not ts_list:
        click.echo(f"No tilt-series found in '{mdoc_dir}'. Exiting.")
        sys.exit(1)

    if ts_filter:
        ts_list = [ts for ts in ts_list if ts in ts_filter]
        if not ts_list:
            click.echo("No tilt-series matched ts_filter in config. Exiting.")
            sys.exit(1)

    click.echo(f"Found {len(ts_list)} tilt-series to process.")

    for ts in ts_list:
        click.echo(f"  Processing {ts} …")
        _make_mdoc_for_ts(ts, mdoc_dir, blended_dir, output_dir, dose_per_tilt, keys_to_copy)

    click.echo("make-mdoc done.")


def _make_mdoc_for_ts(ts, mdoc_dir, blended_dir, output_dir, dose_per_tilt, keys_to_copy):
    """Write the blended mdoc for one tilt-series.

    Logic (mirrors the original make_montages_mdoc_square_frames_mrc.py):
    - Reads the _0_0 corner tile as the metadata template for every tilt angle.
    - Iterates tilts sorted by TiltAngle (acquisition order).
    - ExposureDose = (tilt_index + 0.5) * dose_per_tilt
    - SubFramePath  = blended_dir / {ts}_{tilt_angle}_blended.mrc
    """
    tile_mdoc_paths = sorted(glob.glob(os.path.join(mdoc_dir, f"{ts}_*_*.mrc.mdoc")))
    if not tile_mdoc_paths:
        click.echo(f"  [WARNING] No tile MDOCs found for {ts} — skipping.", err=True)
        return

    # Find the corner (_0_0) tile
    corner_mdoc = None
    for path in tile_mdoc_paths:
        if '_0_0' in os.path.basename(path):
            corner_mdoc = parse_mdoc_file(path)
            break

    if corner_mdoc is None:
        click.echo(f"  [WARNING] _0_0 tile not found for {ts} — skipping.", err=True)
        return

    # Sort z-sections by TiltAngle
    z_values_by_angle = sorted(
        corner_mdoc['z_sections'].keys(),
        key=lambda z: corner_mdoc['z_sections'][z].get('TiltAngle', 0.0),
    )

    out_mdoc = {
        'header':     dict(corner_mdoc['header']),
        'z_sections': {},
    }

    for seq_i, z in enumerate(z_values_by_angle):
        src         = corner_mdoc['z_sections'][z]
        tilt_angle  = src.get('TiltAngle', 0.0)

        new_section = {}

        # Copy requested metadata keys from the corner tile
        for key in keys_to_copy:
            if key in src:
                new_section[key] = src[key]

        # Accumulated dose up to the midpoint of this tilt exposure
        new_section['ExposureDose'] = (seq_i + 0.5) * dose_per_tilt

        # Point SubFramePath at the blended average image produced by 'blend'
        # blend_tiles names files as: {ts}_{tilt_angle}_blended.mrc
        blended_image = os.path.join(blended_dir, f"{ts}_{tilt_angle}_blended.mrc")
        new_section['SubFramePath'] = os.path.abspath(blended_image)

        out_mdoc['z_sections'][seq_i] = new_section

    out_path = os.path.join(output_dir, f"{ts}_blended.mrc.mdoc")
    write_mdoc_file(out_mdoc, out_path)
    click.echo(f"    → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# write-config  (generate a template config file)
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_CONFIG = """\
# =============================================================================
# sqap-montage pipeline config
# =============================================================================
# Run individual steps:
#   python sqap_montage.py crop      --config pipeline.yaml
#   python sqap_montage.py blend     --config pipeline.yaml
#   python sqap_montage.py fill      --config pipeline.yaml
#   python sqap_montage.py make-mdoc --config pipeline.yaml
#
# Generate a fresh copy of this file:
#   python sqap_montage.py write-config pipeline.yaml
# =============================================================================

# Root directory where your data lives.
# All relative paths below are resolved from here.
data_dir: /data1/users/Krios_Data/HRR/HRR036_1_TEM_250220

# =============================================================================
# Global options
# =============================================================================

# Intermediate .mrc files (IMOD stacks, raw blended MRCs) can be large.
# By default they are deleted from the processing/ dirs once each step
# finishes successfully. Text files (.plin, .plout, .txt, logs) are always
# kept — delete the processing/ folder manually if you want to remove those.
# Set to true to keep the intermediate .mrc files too.
keep_intermediate: false

# Number of CPU workers for the crop and blend steps.
# Set to 1 for sequential processing (safest for debugging).
# Set to e.g. 4 or 8 to process multiple tiles / tilt-series in parallel.
num_workers: 1

# =============================================================================
# Step 1: crop — remove blank aperture borders from each tile image
# =============================================================================
crop:
  input_dir:      frames/averages   # motion-corrected average MRCs
  output_dir:     cropped           # creates cropped/averages/ and cropped/frames/
  processing_dir: processing/crop   # boundary coordinate files (can delete after)

  crop_frames:    true              # also crop matching raw frame stacks
  frames_dir:     frames            # raw frame MRCs/TIFs (only if crop_frames: true)
  # Suffix on average filenames absent from frame filenames. Leave "" if stems match.
  # Example: averages "img_avg.mrc" + frames "img.tif" → averages_suffix: "_avg"
  averages_suffix: ""

  crop_x:         3840              # final tile width in pixels
  crop_y:         3840              # final tile height in pixels

  # Detection parameters — usually don't need changing
  filter_window:  200               # moving-average filter width
  mask_threshold: 0.5               # fraction of peak → illuminated region
  trim:           50                # pixels to shave inside the detected edge

# =============================================================================
# Step 2: blend — stitch tiles into one tilt-series per angle
# =============================================================================
blend:
  mdoc_dir:       mdocs             # per-tile SerialEM .mrc.mdoc files
  averages_dir:   cropped/averages  # cropped average images from step 1
  frames_dir:     cropped/frames    # cropped frame stacks from step 1
  output_dir:     blended           # creates blended/averages/ and blended/frames/
  processing_dir: processing        # intermediate stacks, plins, etc. (can delete after)

  blend_size:     11664             # output image edge length after clip resize
  blend_frames:   true              # also blend per-frame stacks
  num_frames:     4                 # raw frames per exposure (used with blend_frames)

  # Process only these tilt-series (leave empty [] for all)
  # ts_filter: [VLP3x3_p01_ts_002, VLP3x3_p01_ts_003]
  ts_filter: []

  # Preview stack — one binned MRC per tilt-series sorted by tilt angle
  preview:         true
  preview_binning: 24          # bin factor passed to newstack -bin
  preview_dir:     blended/previews

  # Optional overrides — both default to siblings of processing_dir
  # (i.e. processing/log/ and processing/sh_files/). Set to a path to relocate.
  # log_dir holds one log per IMOD command, named {ts}_{tilt}_{command}.log,
  # with the executed command, return code, and full stdout + stderr.
  # sh_files_dir holds one re-runnable shell script per tilt-series.
  # log_dir:      processing/log
  # sh_files_dir: processing/sh_files

  # SerialEM occasionally writes PixelShiftFromCenter values that are 1–2 px
  # off the regular grid (e.g. 3682 and 7365 instead of 3682 and 7364).
  # blendmont rejects non-uniform spacings, so each shift is snapped to the
  # nearest integer multiple of the per-axis step before being written to
  # the .plin file. Set to false to feed blendmont the raw mdoc values.
  snap_shifts_to_grid: true

# =============================================================================
# Step 3: fill — fill blending-seam artefacts with local texture
# =============================================================================
fill:
  input_dir:  blended/frames        # blended frame stacks from step 2
  output_dir: blended/frames_filled # gap-filled output
  mask_dir:   blended/frames_masks  # binary gap masks (useful for QC)

  gpus:   "0"                       # comma-separated GPU IDs or "cpu"
  resume: true                      # skip images already in output_dir
  sigma:  5.0                       # gap-detection sensitivity (std devs)
  tile_num: 8                       # grid divisions per axis for local filling

  # Detection kernels [H, W] — tall/wide kernels find horizontal/vertical seams
  detect_kernels:
    - [301, 3]
    - [3, 301]

  # Dilation kernels [H, W] — expand the gap mask before inpainting
  # Applied in both auto-detect and manual-seam modes
  dilate_kernels:
    - [101, 3]
    - [3, 101]
    - [15, 15]

  # Manual seam positions (pixels). Providing these disables auto-detection.
  # Each entry is a single pixel index or an inclusive range "start-end".
  # Leave as [] to use auto-detection.
  #
  #   single pixels:  seam_rows: [3840, 7680]
  #   ranges:         seam_rows: ["3838-3842", "7678-7682"]
  #   mixed:          seam_rows: [3840, "7678-7682"]
  seam_rows: []
  seam_cols: []

# =============================================================================
# Step 4: make-mdoc — build blended .mdoc files from per-tile mdocs
# =============================================================================
make_mdoc:
  mdoc_dir:      mdocs              # same per-tile mdocs as step 2
  blended_dir:   blended/averages   # blended average images from step 2
  output_dir:    blended_mdocs      # output blended .mrc.mdoc files

  dose_per_tilt: 3.0                # electron dose per tilt (e-/Å²)

  # Leave empty [] to process all tilt-series
  ts_filter: []

  # Keys copied from the _0_0 corner tile into the blended mdoc.
  # Uncomment and edit to override the defaults.
  # keys_to_copy:
  #   - MinMaxMean
  #   - StagePosition
  #   - StageZ
  #   - Magnification
  #   - Intensity
  #   - DoseRate
  #   - PixelSpacing
  #   - SpotSize
  #   - ProbeMode
  #   - ImageShift
  #   - RotationAngle
  #   - TiltAngle
  #   - ExposureTime
  #   - Binning
  #   - CameraIndex
  #   - DividedBy2
  #   - OperatingMode
  #   - UsingCDS
  #   - MagIndex
  #   - LowDoseConSet
  #   - CountsPerElectron
  #   - TargetDefocus
  #   - NumSubFrames
  #   - FrameDosesAndNumber
  #   - DateTime
  #   - FilterSlitAndLoss
  #   - UncroppedSize
  #   - RotationAndFlip
  #   - TimeStamp
  #   - SpecimenShift
  #   - EucentricOffset
  #   - Ctfplotter
"""


@cli.command('write-config')
@click.argument('output', default='pipeline.yaml')
def write_config(output):
    """Write a template pipeline.yaml to OUTPUT (default: pipeline.yaml)."""
    if os.path.exists(output):
        click.confirm(f"'{output}' already exists. Overwrite?", abort=True)
    with open(output, 'w') as f:
        f.write(TEMPLATE_CONFIG)
    click.echo(f"Template config written to '{output}'.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cli()
