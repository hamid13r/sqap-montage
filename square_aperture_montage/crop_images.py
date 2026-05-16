#!/usr/bin/env python3
"""
crop_images.py — Crop the blank border outside the square aperture from each tile image.

SerialEM collects each tile on a detector larger than the illuminated square
aperture, leaving dark/empty borders. This script detects those borders using
intensity profiles, crops them away, and optionally applies the same crop to
the corresponding raw frame stacks.

Typical usage
-------------
  sam-crop --input-dir frames/averages --output-dir cropped --frames-dir frames

  sam-crop --input-dir frames/averages/img.mrc --output-dir cropped --no-crop-frames

Run ``sam-crop --help`` for all options.
"""

import glob
import os

import click
import mrcfile
import numpy as np
from PIL import Image
from tqdm import tqdm


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
                          x_start, x_end, y_start, y_end):
    """Apply pre-computed crop boundaries to the frame stack for one exposure."""
    os.makedirs(output_frames_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    frame_mrc = os.path.join(frames_dir, f"{stem}.mrc")
    frame_tif = os.path.join(frames_dir, f"{stem}.tif")

    if os.path.exists(frame_mrc):
        with mrcfile.open(frame_mrc, mode='r') as mrc:
            frames_data = mrc.data.copy()
        cropped = frames_data[:, y_start:y_end, x_start:x_end]
        with mrcfile.new(os.path.join(output_frames_dir, f"{stem}.mrc"), overwrite=True) as mrc_out:
            mrc_out.set_data(cropped)

    elif os.path.exists(frame_tif):
        tif = Image.open(frame_tif)
        frames_data = []
        for i in range(tif.n_frames):
            tif.seek(i)
            frames_data.append(np.flipud(np.array(tif)))
        frames_data = np.array(frames_data)
        cropped = frames_data[:, y_start:y_end, x_start:x_end]
        with mrcfile.new(os.path.join(output_frames_dir, f"{stem}.mrc"), overwrite=True) as mrc_out:
            mrc_out.set_data(cropped)

    else:
        print(f"  [WARNING] No frame file found for {stem} in {frames_dir}")


@click.command()
@click.option('--input-dir',       default='frames/averages', show_default=True,
              help='Directory of input average MRCs, or path to a single MRC.')
@click.option('--output-dir',      default='cropped',         show_default=True,
              help='Root output directory. Averages → OUTPUT/averages/, frames → OUTPUT/frames/.')
@click.option('--processing-dir',  default='processing/crop', show_default=True,
              help='Directory for boundary coordinate files.')
@click.option('--filter-window',   default=200,  show_default=True,
              help='Moving-average filter width for intensity profile smoothing.')
@click.option('--mask-threshold',  default=0.5,  show_default=True,
              help='Fraction of peak intensity defining the illuminated region.')
@click.option('--trim',            default=50,   show_default=True,
              help='Pixels to trim inside the detected boundary edge.')
@click.option('--crop-x',          default=3840, show_default=True,
              help='Final crop width in pixels.')
@click.option('--crop-y',          default=3840, show_default=True,
              help='Final crop height in pixels.')
@click.option('--crop-frames/--no-crop-frames', default=True, show_default=True,
              help='Also crop matching frame stacks from --frames-dir.')
@click.option('--frames-dir',      default='frames', show_default=True,
              help='Directory containing raw frame MRC/TIF files.')
def main(input_dir, output_dir, processing_dir, filter_window, mask_threshold,
         trim, crop_x, crop_y, crop_frames, frames_dir):
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

    for image_file in tqdm(image_files, desc="Cropping"):
        x_start, x_end, y_start, y_end = crop_average(
            image_file, output_averages_dir, processing_dir,
            filter_size=filter_window, mask_threshold=mask_threshold,
            trim=trim, crop_x=crop_x, crop_y=crop_y,
        )
        if crop_frames:
            crop_frames_for_image(image_file, frames_dir, output_frames_dir,
                                  x_start, x_end, y_start, y_end)

    print("Done.")


if __name__ == '__main__':
    main()
