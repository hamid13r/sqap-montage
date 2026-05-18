#!/usr/bin/env python3
"""
crop_images.py — Crop the blank border outside the square aperture from each tile image.

SerialEM collects each tile on a detector larger than the illuminated square
aperture, leaving dark/empty borders. This script detects those borders using
intensity profiles, crops them away, and optionally applies the same crop to
the corresponding raw frame stacks.

Frame files may be MRC or TIFF (single- or multi-page). TIFF files are
converted to MRC using IMOD ``tif2mrc`` before cropping.

If the motion-corrected averages carry an extra suffix that the raw frame
files do not (e.g. averages are ``img_avg.mrc`` while frames are ``img.tif``),
supply ``--averages-suffix _avg`` so the frame lookup strips it correctly.

Typical usage
-------------
  sam-crop --input-dir frames/averages --output-dir cropped --frames-dir frames

  sam-crop --averages-suffix _avg --input-dir frames/averages --output-dir cropped

Run ``sam-crop --help`` for all options.
"""

import glob
import os
import subprocess

import click
import mrcfile
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# TIF → MRC conversion
# ---------------------------------------------------------------------------

def tif_to_mrc(tif_path: str, mrc_path: str) -> bool:
    """Convert a TIFF file to MRC using IMOD ``tif2mrc``.

    Handles both single-image and multi-page (frame stack) TIFFs.
    See https://bio3d.colorado.edu/imod/doc/man/tif2mrc.html

    Returns True on success, False if the conversion fails (a warning is
    printed in that case).
    """
    result = subprocess.run(
        ['tif2mrc', tif_path, mrc_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print(f"  [WARNING] tif2mrc failed for {os.path.basename(tif_path)}:\n"
              f"  {result.stderr.decode().strip()}")
        return False
    return True


# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------

def detect_crop_boundaries(mrc_image, filter_size=200, mask_threshold=0.5,
                            trim=50, crop_x=3840, crop_y=3840):
    """Detect crop boundaries for a 2-D MRC image array.

    Uses moving-average intensity profiles to find the edges of the illuminated
    area, trims extra fringe pixels, then returns a centred crop of crop_x × crop_y.

    Returns
    -------
    tuple of (x_start, x_end, y_start, y_end)
    """
    x_profile = np.convolve(
        np.sum(mrc_image, axis=0) / np.max(np.sum(mrc_image, axis=0)),
        np.ones(filter_size) / filter_size, mode='same',
    )
    y_profile = np.convolve(
        np.sum(mrc_image, axis=1) / np.max(np.sum(mrc_image, axis=1)),
        np.ones(filter_size) / filter_size, mode='same',
    )

    x_lit = np.where((x_profile / np.max(x_profile)) > mask_threshold)[0]
    y_lit = np.where((y_profile / np.max(y_profile)) > mask_threshold)[0]

    center_x = (x_lit[0] + trim + x_lit[-1] - trim) // 2
    center_y = (y_lit[0] + trim + y_lit[-1] - trim) // 2

    x_start = center_x - crop_x // 2
    x_end   = center_x + crop_x // 2
    y_start = center_y - crop_y // 2
    y_end   = center_y + crop_y // 2

    return x_start, x_end, y_start, y_end


# ---------------------------------------------------------------------------
# Per-image crop functions
# ---------------------------------------------------------------------------

def crop_average(image_path, output_averages_dir, processing_dir,
                 filter_size=200, mask_threshold=0.5, trim=50,
                 crop_x=3840, crop_y=3840):
    """Crop one motion-corrected average MRC and save boundary coordinates."""
    os.makedirs(output_averages_dir, exist_ok=True)
    os.makedirs(processing_dir, exist_ok=True)

    image_name = os.path.basename(image_path)
    stem = os.path.splitext(image_name)[0]

    with mrcfile.open(image_path, mode='r') as mrc:
        mrc_image = mrc.data.copy()

    x_start, x_end, y_start, y_end = detect_crop_boundaries(
        mrc_image, filter_size, mask_threshold, trim, crop_x, crop_y
    )

    cropped = mrc_image[y_start:y_end, x_start:x_end]
    with mrcfile.new(os.path.join(output_averages_dir, image_name), overwrite=True) as mrc_out:
        mrc_out.set_data(cropped)

    boundary_file = os.path.join(processing_dir, f"{stem}_crop_boundaries.txt")
    with open(boundary_file, 'w') as f:
        f.write("x_start,x_end,y_start,y_end\n")
        f.write(f"{x_start},{x_end},{y_start},{y_end}\n")

    return x_start, x_end, y_start, y_end


def crop_frames_for_image(image_path, frames_dir, output_frames_dir,
                          x_start, x_end, y_start, y_end,
                          averages_suffix=''):
    """Apply pre-computed crop boundaries to the frame stack for one exposure.

    Parameters
    ----------
    image_path : str
        Path to the motion-corrected average MRC (used to derive the stem).
    frames_dir : str
        Directory containing raw frame stacks (.mrc, .tif, or .tiff).
    output_frames_dir : str
        Directory where the cropped frame MRC will be written.
    x_start, x_end, y_start, y_end : int
        Crop boundaries from ``crop_average``.
    averages_suffix : str
        Suffix on the average filename that is absent from the frame filename.
        Example: if the average is ``img_avg.mrc`` and the frame is ``img.tif``,
        pass ``averages_suffix='_avg'`` so the lookup strips it correctly.
        Leave as '' when average and frame stems match exactly.

    Frame format handling
    ---------------------
    - **.mrc** — opened directly with mrcfile.
    - **.tif / .tiff** — converted to MRC with IMOD ``tif2mrc`` (which
      correctly handles multi-page cryo-EM frame stacks), then cropped.
    """
    os.makedirs(output_frames_dir, exist_ok=True)

    avg_stem = os.path.splitext(os.path.basename(image_path))[0]

    # Strip the averages-only suffix to get the base frame stem
    if averages_suffix and avg_stem.endswith(averages_suffix):
        frame_stem = avg_stem[:-len(averages_suffix)]
    else:
        frame_stem = avg_stem

    # Candidate frame paths in preference order
    frame_mrc  = os.path.join(frames_dir, f"{frame_stem}.mrc")
    frame_tif  = os.path.join(frames_dir, f"{frame_stem}.tif")
    frame_tiff = os.path.join(frames_dir, f"{frame_stem}.tiff")

    out_mrc = os.path.join(output_frames_dir, f"{frame_stem}.mrc")

    if os.path.exists(frame_mrc):
        with mrcfile.open(frame_mrc, mode='r') as mrc:
            frames_data = mrc.data.copy()
        cropped = frames_data[:, y_start:y_end, x_start:x_end]
        with mrcfile.new(out_mrc, overwrite=True) as mrc_out:
            mrc_out.set_data(cropped)

    elif os.path.exists(frame_tif) or os.path.exists(frame_tiff):
        tif_path = frame_tif if os.path.exists(frame_tif) else frame_tiff
        # Convert to the output location with tif2mrc, then crop in place
        if tif_to_mrc(tif_path, out_mrc):
            with mrcfile.open(out_mrc, mode='r') as mrc:
                frames_data = mrc.data.copy()
            cropped = frames_data[:, y_start:y_end, x_start:x_end]
            with mrcfile.new(out_mrc, overwrite=True) as mrc_out:
                mrc_out.set_data(cropped)

    else:
        print(f"  [WARNING] No frame file found for '{frame_stem}' in {frames_dir}")


# ---------------------------------------------------------------------------
# Parallel worker (must be at module level to be picklable)
# ---------------------------------------------------------------------------

def _crop_image_worker(args):
    """Top-level worker for ProcessPoolExecutor.

    ``args`` is a tuple matching the call in ``sqap_montage.py``:
    (image_file, output_averages_dir, processing_dir, output_frames_dir,
     frames_dir, crop_frames, averages_suffix,
     filter_window, mask_threshold, trim, crop_x, crop_y)
    """
    (image_file, output_averages_dir, processing_dir, output_frames_dir,
     frames_dir, crop_frames, averages_suffix,
     filter_window, mask_threshold, trim, crop_x, crop_y) = args

    x0, x1, y0, y1 = crop_average(
        image_file, output_averages_dir, processing_dir,
        filter_size=filter_window, mask_threshold=mask_threshold,
        trim=trim, crop_x=crop_x, crop_y=crop_y,
    )
    if crop_frames:
        crop_frames_for_image(
            image_file, frames_dir, output_frames_dir,
            x0, x1, y0, y1, averages_suffix=averages_suffix,
        )
    return os.path.basename(image_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option('--input-dir',        default='frames/averages', show_default=True,
              help='Directory of input average MRCs, or path to a single MRC.')
@click.option('--output-dir',       default='cropped',         show_default=True,
              help='Root output directory. Averages → OUTPUT/averages/, frames → OUTPUT/frames/.')
@click.option('--processing-dir',   default='processing/crop', show_default=True,
              help='Directory for boundary coordinate files.')
@click.option('--filter-window',    default=200,  show_default=True,
              help='Moving-average filter width for intensity profile smoothing.')
@click.option('--mask-threshold',   default=0.5,  show_default=True,
              help='Fraction of peak intensity defining the illuminated region.')
@click.option('--trim',             default=50,   show_default=True,
              help='Pixels to trim inside the detected boundary edge.')
@click.option('--crop-x',           default=3840, show_default=True,
              help='Final crop width in pixels.')
@click.option('--crop-y',           default=3840, show_default=True,
              help='Final crop height in pixels.')
@click.option('--crop-frames/--no-crop-frames', default=True, show_default=True,
              help='Also crop matching frame stacks from --frames-dir.')
@click.option('--frames-dir',       default='frames', show_default=True,
              help='Directory containing raw frame MRC/TIF files.')
@click.option('--averages-suffix',  default='', show_default=True,
              help='Suffix on average filenames absent from frame filenames '
                   '(e.g. "_avg"). Stripped when looking up the matching frame.')
def main(input_dir, output_dir, processing_dir, filter_window, mask_threshold,
         trim, crop_x, crop_y, crop_frames, frames_dir, averages_suffix):
    """Crop the dark border outside the square aperture from tile images."""
    output_averages_dir = os.path.join(output_dir, 'averages')
    output_frames_dir   = os.path.join(output_dir, 'frames')

    if os.path.isdir(input_dir):
        image_files = sorted(glob.glob(os.path.join(input_dir, '*.mrc')))
    elif os.path.isfile(input_dir):
        image_files = [input_dir]
    else:
        raise click.BadParameter(f"'{input_dir}' is neither a file nor a directory.",
                                 param_hint='--input-dir')

    if not image_files:
        print(f"No .mrc files found in '{input_dir}'. Exiting.")
        return

    print(f"Found {len(image_files)} image(s) to process.")
    if averages_suffix:
        print(f"Averages suffix: '{averages_suffix}' (stripped when looking up frames)")

    for image_file in tqdm(image_files, desc="Cropping"):
        x_start, x_end, y_start, y_end = crop_average(
            image_file, output_averages_dir, processing_dir,
            filter_size=filter_window, mask_threshold=mask_threshold,
            trim=trim, crop_x=crop_x, crop_y=crop_y,
        )
        if crop_frames:
            crop_frames_for_image(image_file, frames_dir, output_frames_dir,
                                  x_start, x_end, y_start, y_end,
                                  averages_suffix=averages_suffix)

    print("Done.")


if __name__ == '__main__':
    main()
