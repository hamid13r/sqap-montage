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
    cfg      = load_config(config)
    c        = cfg.get('crop', {})
    data_dir = cfg.get('data_dir', '.')

    input_dir      = resolve(data_dir, c.get('input_dir',      'frames/averages'))
    output_dir     = resolve(data_dir, c.get('output_dir',     'cropped'))
    processing_dir = resolve(data_dir, c.get('processing_dir', 'processing/crop'))
    frames_dir     = resolve(data_dir, c.get('frames_dir',     'frames'))
    crop_frames    = c.get('crop_frames',    True)
    crop_x         = c.get('crop_x',         3840)
    crop_y         = c.get('crop_y',         3840)
    filter_window  = c.get('filter_window',  200)
    mask_threshold = c.get('mask_threshold', 0.5)
    trim           = c.get('trim',           50)

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

    for image_file in tqdm(image_files, desc="Cropping"):
        x0, x1, y0, y1 = crop_average(
            image_file, output_averages_dir, processing_dir,
            filter_size=filter_window, mask_threshold=mask_threshold,
            trim=trim, crop_x=crop_x, crop_y=crop_y,
        )
        if crop_frames:
            crop_frames_for_image(image_file, frames_dir, output_frames_dir, x0, x1, y0, y1)

    click.echo("Crop done.")


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
    cfg      = load_config(config)
    b        = cfg.get('blend', {})
    data_dir = cfg.get('data_dir', '.')

    mdoc_dir       = resolve(data_dir, b.get('mdoc_dir',       'mdocs'))
    averages_dir   = resolve(data_dir, b.get('averages_dir',   'cropped/averages'))
    frames_dir     = resolve(data_dir, b.get('frames_dir',     'cropped/frames'))
    output_dir     = resolve(data_dir, b.get('output_dir',     'blended'))
    processing_dir = resolve(data_dir, b.get('processing_dir', 'processing'))
    blend_size     = b.get('blend_size',   11664)
    blend_frames   = b.get('blend_frames', True)
    num_frames     = b.get('num_frames',   4)
    ts_filter      = b.get('ts_filter',    [])

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

    for i, ts in enumerate(ts_list):
        click.echo(f"\n[{i + 1}/{len(ts_list)}] {ts}")
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
        )

    click.echo("\nBlend done.")


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

    from square_aperture_montage.remove_gaps import process_image, _process_with_gpu

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

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir,   exist_ok=True)

    image_list = sorted(glob.glob(os.path.join(input_dir, '*.mrc')))
    if not image_list:
        click.echo(f"No .mrc files found in '{input_dir}'. Exiting.")
        sys.exit(1)

    click.echo(f"Found {len(image_list)} images to process.")

    if gpus.strip().lower() == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
        for img in tqdm(image_list, desc="Filling gaps"):
            process_image(img, output_dir, mask_dir, device, resume, sigma, tile_num)
        click.echo("Fill done.")
        return

    available = torch.cuda.device_count()
    gpu_ids   = [int(g.strip()) for g in gpus.split(',') if g.strip().isdigit()]
    gpu_ids   = [g for g in gpu_ids if g < available] or [0]

    click.echo(f"Using GPUs: {gpu_ids}")
    task_args = [
        (img, gpu_ids[i % len(gpu_ids)], output_dir, mask_dir, resume, sigma, tile_num)
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
# =============================================================================

# Root directory where your data lives.
# All relative paths below are resolved from here.
data_dir: /data1/users/Krios_Data/HRR/HRR036_1_TEM_250220

# =============================================================================
# Step 1: crop — remove blank aperture borders from each tile image
# =============================================================================
crop:
  input_dir:      frames/averages   # motion-corrected average MRCs
  output_dir:     cropped           # creates cropped/averages/ and cropped/frames/
  processing_dir: processing/crop   # boundary coordinate files (can delete after)

  crop_frames:    true              # also crop matching raw frame stacks
  frames_dir:     frames            # raw frame MRCs/TIFs (only if crop_frames: true)

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
