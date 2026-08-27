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
from typing import NamedTuple

import click
import mrcfile
import numpy as np
from tqdm import tqdm


class CropBounds(NamedTuple):
    """Result of :func:`detect_crop_boundaries`.

    The first four fields are the crop window actually used for slicing; the
    remainder are diagnostics written to the boundary QC file so a systematic
    crop offset is visible at a glance without re-running detection.
    """
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    x_min: int          # detected illumination extent (left / right columns)
    x_max: int
    y_min: int          # detected illumination extent (top / bottom rows)
    y_max: int
    center_x: float     # illumination center the window was placed on
    center_y: float
    shifted: bool       # True when the window was translated to fit the image


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
                            crop_x=3840, crop_y=3840):
    """Detect a crop_x × crop_y window centered on the illuminated aperture.

    Uses moving-average intensity profiles to find the edges of the illuminated
    area, then returns a ``crop_x × crop_y`` window centered on that area. The
    output is *always* exactly ``crop_x × crop_y``: if the centered window would
    spill past an image edge it is translated back inside the image (not
    truncated), and ``shifted`` is set on the returned :class:`CropBounds`.

    ``mask_threshold`` is a fraction of each profile's own range (min..max),
    not of its absolute peak — motion-corrected averages carry a large
    constant dose background, so the illuminated region is often under 1%
    brighter than the dark border, and a peak-relative threshold would never
    exclude anything.

    Returns
    -------
    CropBounds
        Crop window plus detection diagnostics.

    Raises
    ------
    ValueError
        If the image has no positive signal, no region exceeds
        ``mask_threshold`` (no aperture detected), or the requested crop is
        larger than the image (a crop that does not fit is a hard error, not
        silently undersized — blendmont requires uniform piece sizes).
    """
    # Cast to float64 once. MRC averages are often int16/float16; summing a
    # 4096-wide row in the native dtype overflows and poisons the profile
    # with NaN/garbage values.
    img = np.asarray(mrc_image, dtype=np.float64)

    col_sum = np.sum(img, axis=0)
    row_sum = np.sum(img, axis=1)

    col_max = col_sum.max() if col_sum.size else 0.0
    row_max = row_sum.max() if row_sum.size else 0.0
    if (not np.isfinite(col_max) or not np.isfinite(row_max)
            or col_max <= 0 or row_max <= 0):
        raise ValueError(
            "Image has no positive signal — cannot detect aperture boundaries."
        )

    def smooth(profile, width):
        # Edge-replicate before convolving instead of relying on 'same' mode's
        # implicit zero-padding, which pulls the first/last width/2 samples of
        # the moving average toward zero regardless of the real signal there
        # and corrupts exactly the boundary region this function is trying to
        # locate.
        pad = width // 2
        padded = np.pad(profile, pad, mode='edge')
        return np.convolve(padded, np.ones(width) / width, mode='valid')[:profile.size]

    x_profile = smooth(col_sum, filter_size)
    y_profile = smooth(row_sum, filter_size)

    # Threshold relative to each profile's own dynamic range (min..max), not
    # its absolute peak. Motion-corrected averages carry a large constant dose
    # background — the illuminated aperture can be less than 1% brighter than
    # the dark border — so peak-relative thresholding (profile/peak >
    # mask_threshold) is satisfied everywhere and "detection" silently
    # degenerates to the full frame every time.
    def lit_extent(profile):
        lo, hi = profile.min(), profile.max()
        if hi <= lo:
            return np.empty(0, dtype=int)
        thr = lo + mask_threshold * (hi - lo)
        return np.where(profile > thr)[0]

    x_lit = lit_extent(x_profile)
    y_lit = lit_extent(y_profile)

    if x_lit.size == 0 or y_lit.size == 0:
        raise ValueError(
            f"No region above mask_threshold={mask_threshold} found. "
            "Try lowering --mask-threshold or check the input image."
        )

    # Detected illumination extent.
    x_min, x_max = int(x_lit[0]), int(x_lit[-1])
    y_min, y_max = int(y_lit[0]), int(y_lit[-1])

    height, width = img.shape[-2:]
    height, width = int(height), int(width)

    # A crop larger than the image can never be produced at the requested size.
    # Fail loudly rather than emit an undersized tile — downstream blendmont
    # requires every piece to be exactly the same size.
    if crop_x > width or crop_y > height:
        raise ValueError(
            f"Requested crop {crop_x}×{crop_y} does not fit inside image "
            f"{width}×{height}. Reduce --crop-x / --crop-y."
        )

    # Center of the illuminated area. Keep as float and round once, so a
    # half-pixel center does not bias the window toward lower x/y.
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # Cut a crop_x × crop_y window centered on that point. Derive each end from
    # start + size (not center + size/2) so the window is exactly crop_x/crop_y
    # wide even for odd crop sizes.
    x_start = int(round(center_x - crop_x / 2))
    x_end   = x_start + crop_x
    y_start = int(round(center_y - crop_y / 2))
    y_end   = y_start + crop_y

    # If the window spills past an edge, translate it back inside the image
    # rather than truncating it. The output must always be exactly
    # crop_x × crop_y; only its position changes.
    ideal_x_start, ideal_y_start = x_start, y_start
    shifted = False
    if x_start < 0:
        x_start, x_end = 0, crop_x
        shifted = True
    elif x_end > width:
        x_start, x_end = width - crop_x, width
        shifted = True
    if y_start < 0:
        y_start, y_end = 0, crop_y
        shifted = True
    elif y_end > height:
        y_start, y_end = height - crop_y, height
        shifted = True

    if shifted:
        dx = x_start - ideal_x_start
        dy = y_start - ideal_y_start
        print(
            f"  [WARNING] crop window translated by (dx={dx:+d}, dy={dy:+d}) px "
            f"to stay inside the image: illumination center "
            f"({center_x:.1f}, {center_y:.1f}) is no longer centered in the "
            f"{crop_x}×{crop_y} crop  x[{x_start}:{x_end}] y[{y_start}:{y_end}] "
            f"(image {width}×{height})"
        )

    return CropBounds(x_start, x_end, y_start, y_end,
                      x_min, x_max, y_min, y_max,
                      center_x, center_y, shifted)


# ---------------------------------------------------------------------------
# Per-image crop functions
# ---------------------------------------------------------------------------

def crop_average(image_path, output_averages_dir, processing_dir,
                 filter_size=200, mask_threshold=0.5,
                 crop_x=3840, crop_y=3840):
    """Crop one motion-corrected average MRC and save boundary coordinates."""
    os.makedirs(output_averages_dir, exist_ok=True)
    os.makedirs(processing_dir, exist_ok=True)

    image_name = os.path.basename(image_path)
    stem = os.path.splitext(image_name)[0]

    with mrcfile.open(image_path, mode='r') as mrc:
        mrc_image = mrc.data.copy()

    try:
        b = detect_crop_boundaries(
            mrc_image, filter_size, mask_threshold, crop_x, crop_y
        )
    except ValueError as e:
        # Attach the offending file path so parallel workers produce a
        # message you can actually act on.
        raise ValueError(f"{image_path}: {e}") from e

    cropped = mrc_image[b.y_start:b.y_end, b.x_start:b.x_end]
    with mrcfile.new(os.path.join(output_averages_dir, image_name), overwrite=True) as mrc_out:
        mrc_out.set_data(cropped)

    boundary_file = os.path.join(processing_dir, f"{stem}_crop_boundaries.txt")
    with open(boundary_file, 'w') as f:
        f.write("x_start,x_end,y_start,y_end,x_min,x_max,y_min,y_max,"
                "center_x,center_y,shifted\n")
        f.write(f"{b.x_start},{b.x_end},{b.y_start},{b.y_end},"
                f"{b.x_min},{b.x_max},{b.y_min},{b.y_max},"
                f"{b.center_x},{b.center_y},{b.shifted}\n")

    return b.x_start, b.x_end, b.y_start, b.y_end


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
     filter_window, mask_threshold, crop_x, crop_y)
    """
    (image_file, output_averages_dir, processing_dir, output_frames_dir,
     frames_dir, crop_frames, averages_suffix,
     filter_window, mask_threshold, crop_x, crop_y) = args

    x0, x1, y0, y1 = crop_average(
        image_file, output_averages_dir, processing_dir,
        filter_size=filter_window, mask_threshold=mask_threshold,
        crop_x=crop_x, crop_y=crop_y,
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
              help='Fraction between each profile\'s min and max intensity '
                   'defining the illuminated region.')
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
         crop_x, crop_y, crop_frames, frames_dir, averages_suffix):
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
            crop_x=crop_x, crop_y=crop_y,
        )
        if crop_frames:
            crop_frames_for_image(image_file, frames_dir, output_frames_dir,
                                  x_start, x_end, y_start, y_end,
                                  averages_suffix=averages_suffix)

    print("Done.")


if __name__ == '__main__':
    main()
