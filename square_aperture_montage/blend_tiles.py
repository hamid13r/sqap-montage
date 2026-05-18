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
import os
import sys
from pathlib import Path, PureWindowsPath
import subprocess

import click
import tqdm

from .mdoc_reader import parse_mdoc_file, write_mdoc_file


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


def imod_newstack(image_list, frame_num, stack_out, processing_dir):
    """Stack images with IMOD newstack."""
    stem = Path(stack_out).stem
    filein = os.path.join(processing_dir, f"{stem}_{frame_num}.filein")
    write_fileinlist(image_list, frame_num, filein)
    result = subprocess.run(
        f"newstack -filein {filein} -output {stack_out}",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
    )
    if result.returncode != 0:
        print(f"  [WARNING] newstack failed for {stack_out}:\n"
              f"  {result.stderr.decode().strip()}")
    return result


def imod_blendmont(stk_file, plin_file, plout_file, blend_size,
                   blended_output, processing_dir):
    """Blend a montage stack and resize with IMOD blendmont + clip."""
    rootname = Path(blended_output).stem
    intermediate = os.path.join(processing_dir, f"{rootname}_raw.mrc")

    result_blend = subprocess.run(
        f"blendmont -imin {stk_file} -plin {plin_file} "
        f"-imout {intermediate} "
        f"-roo {os.path.join(processing_dir, rootname)} "
        f"-al {plout_file} -adj -shift",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
    )
    if result_blend.returncode != 0:
        print(f"  [WARNING] blendmont failed:\n  {result_blend.stderr.decode().strip()}")

    result_clip = subprocess.run(
        f"clip resize -ox {blend_size} -oy {blend_size} {intermediate} {blended_output}",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
    )
    if result_clip.returncode != 0:
        print(f"  [WARNING] clip failed:\n  {result_clip.stderr.decode().strip()}")

    return result_blend, result_clip


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
     blend_size, blend_frames, num_frames)

    Returns
    -------
    tuple of (tilt_i, tilt_angle, blended_avg_path, blended_frames_path_or_None)
    """
    (tilt_i, tile_sections,
     ts, cropped_averages_abs, cropped_frames_abs,
     processing_averages_dir, processing_frames_dir,
     output_averages_dir, output_frames_dir,
     blend_size, blend_frames, num_frames) = args

    shifts_list = []
    image_list  = []
    tilt_angle  = tilt_i   # fallback

    for section in tile_sections:
        tilt_angle = section.get('TiltAngle', tilt_i)
        shifts     = section.get('PixelShiftFromCenter', [0, 0])
        shifts_list.append([int(shifts[0]), int(shifts[1]), 0])
        subframe   = PureWindowsPath(section.get('SubFramePath', ''))
        image_list.append(
            os.path.join(cropped_averages_abs, subframe.name.replace('.tif', '.mrc'))
        )

    plin_file   = os.path.join(processing_averages_dir, f"{ts}_{tilt_angle}.plin")
    plout_file  = os.path.join(processing_averages_dir, f"{ts}_{tilt_angle}.plout")
    stack_file  = os.path.join(processing_averages_dir, f"{ts}_{tilt_angle}.mrc")
    blended_out = os.path.join(output_averages_dir,     f"{ts}_{tilt_angle}_blended.mrc")

    write_plin(shifts_list, plin_file)
    imod_newstack(image_list, 0, stack_file, processing_averages_dir)
    imod_blendmont(stack_file, plin_file, plout_file, blend_size,
                   blended_out, processing_averages_dir)

    frame_stack_out = None
    if blend_frames:
        frame_output_list = []
        for frame_i in range(num_frames):
            frame_image_list  = []
            frame_shifts_list = []
            for section in tile_sections:
                shifts   = section.get('PixelShiftFromCenter', [0, 0])
                frame_shifts_list.append([int(shifts[0]), int(shifts[1]), 0])
                subframe = PureWindowsPath(section.get('SubFramePath', ''))
                frame_image_list.append(
                    os.path.join(cropped_frames_abs, subframe.name.replace('.tif', '.mrc'))
                )
            frame_stack   = os.path.join(processing_frames_dir, f"{ts}_{tilt_angle}_frame{frame_i}.mrc")
            frame_plin    = os.path.join(processing_frames_dir, f"{ts}_{tilt_angle}_frame{frame_i}.plin")
            frame_plout   = os.path.join(processing_frames_dir, f"{ts}_{tilt_angle}_frame{frame_i}.plout")
            frame_blended = os.path.join(processing_frames_dir, f"{ts}_{tilt_angle}_frame{frame_i}_blended.mrc")

            write_plin(frame_shifts_list, frame_plin)
            imod_newstack(frame_image_list, frame_i, frame_stack, processing_frames_dir)
            imod_blendmont(frame_stack, frame_plin, frame_plout, blend_size,
                           frame_blended, processing_frames_dir)
            frame_output_list.append(os.path.abspath(frame_blended))

        frame_stack_out = os.path.join(output_frames_dir, f"{ts}_{tilt_angle}_blended_frames.mrc")
        imod_newstack(frame_output_list, 0, frame_stack_out, processing_frames_dir)

    return (tilt_i, tilt_angle,
            os.path.abspath(blended_out),
            os.path.abspath(frame_stack_out) if frame_stack_out else None)


# ---------------------------------------------------------------------------
# Main tilt-series processing function
# ---------------------------------------------------------------------------

def process_tilt_series(ts, mdoc_dir, cropped_averages_dir, cropped_frames_dir,
                        processing_averages_dir, processing_frames_dir,
                        output_averages_dir, output_frames_dir,
                        output_averages_mdoc_dir, output_frames_mdoc_dir,
                        blend_size, blend_frames, num_frames,
                        num_workers=1, show_progress=True, tqdm_position=0):
    """Blend all tiles for one tilt-series.

    Parameters
    ----------
    num_workers : int
        Number of parallel workers for per-tilt blending.
        1 = sequential (default). >1 = parallel via ProcessPoolExecutor.
    show_progress : bool
        Show a per-tilt tqdm progress bar.
    tqdm_position : int
        tqdm ``position`` for the inner bar (use 1 when an outer bar is at 0).
    """
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
         blend_size, blend_frames, num_frames)
        for tilt_i in range(num_tilts)
    ]

    if show_progress:
        pbar = tqdm.tqdm(total=num_tilts,
                         desc=f"  {ts}",
                         position=tqdm_position,
                         leave=True)
    else:
        pbar = None

    results = {}   # tilt_i → (tilt_angle, blended_avg, blended_frames)

    if num_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_blend_tilt_worker, a): a[0] for a in task_args}
            for future in concurrent.futures.as_completed(futures):
                tilt_i, tilt_angle, avg_path, frm_path = future.result()
                results[tilt_i] = (tilt_angle, avg_path, frm_path)
                if pbar is not None:
                    pbar.set_postfix(angle=f"{tilt_angle:+.1f}°")
                    pbar.update(1)
    else:
        for args in task_args:
            tilt_i, tilt_angle, avg_path, frm_path = _blend_tilt_worker(args)
            results[tilt_i] = (tilt_angle, avg_path, frm_path)
            if pbar is not None:
                pbar.set_postfix(angle=f"{tilt_angle:+.1f}°")
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    # Update mdoc with output paths (must happen after all workers finish)
    for tilt_i, (tilt_angle, avg_path, frm_path) in results.items():
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
@click.option('--ts', 'ts_filter', default=None, multiple=True,
              help='Process only these tilt-series names (repeatable). Defaults to all.')
def main(mdoc_dir, averages_dir, frames_dir, output_dir, processing_dir,
         blend_size, blend_frames, num_frames, ts_filter):
    """Blend 3×3 montage tile images into a single giant tilt-series."""
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
        ts_list = [ts for ts in ts_list if ts in ts_filter]
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
        )

    print("\nDone.")


if __name__ == '__main__':
    main()
