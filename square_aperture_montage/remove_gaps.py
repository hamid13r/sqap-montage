#!/usr/bin/env python3
"""
remove_gaps.py — Fill blending-seam gaps in blended MRC frame stacks.

Two masking modes:

Auto-detect (default)
    Uses GPU-accelerated gradient and intensity-outlier analysis to find
    seam artefacts automatically.

Manual seams
    The user supplies explicit row and/or column indices.  The mask is built
    directly from those positions and then dilated — no GPU detection needed.
    Useful when you know exactly where the tile boundaries land in the image.

The mask is filled by drawing random samples from surrounding tissue inside
the same local tile (no interpolation, preserves noise statistics).

Typical usage
-------------
  # auto-detect
  sam-fill --input-dir blended/frames --output-dir blended/frames_filled --gpus 0,1,2

  # manual seams at rows/cols 3840 and 7680 (3×3 grid with 3840-px tiles)
  sam-fill --seam-row 3840 --seam-row 7680 --seam-col 3840 --seam-col 7680 --gpus cpu

Run ``sam-fill --help`` for all options.
"""

import concurrent.futures
import glob
import os
import sys

import click
import mrcfile
import numpy as np
import tqdm

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cc3d
    CC3D_AVAILABLE = True
except ImportError:
    CC3D_AVAILABLE = False


def write_mrc(output_path, img):
    """Write a numpy array to an MRC file."""
    with mrcfile.new(output_path, overwrite=True) as mrc_out:
        mrc_out.set_data(img)
        mrc_out.update_header_from_data()


def remove_small_objects(mask_np, min_size=200):
    """Remove connected components smaller than min_size pixels."""
    if not CC3D_AVAILABLE:
        return mask_np
    labeled = cc3d.connected_components(mask_np, connectivity=4)
    unique, counts = np.unique(labeled, return_counts=True)
    small = unique[(counts < min_size) & (unique != 0)]
    mask_np[np.isin(labeled, small)] = 0
    return mask_np


def build_gap_mask(mrc_tensor, device, detect_kernels=None, dilate_kernels=None, sigma=5.0):
    """Detect seam/gap regions as a binary mask using gradient + intensity outliers."""
    if detect_kernels is None:
        detect_kernels = [(301, 3), (3, 301)]
    if dilate_kernels is None:
        dilate_kernels = [(101, 3), (3, 101), (15, 15)]

    projected = torch.mean(mrc_tensor, dim=0)
    batch     = projected.unsqueeze(0).unsqueeze(0)

    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                       dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    grad_mag = torch.sqrt(
        torch.nn.functional.conv2d(batch, kx, padding='same')[0, 0] ** 2 +
        torch.nn.functional.conv2d(batch, ky, padding='same')[0, 0] ** 2
    )

    mask_out = torch.zeros_like(projected, dtype=torch.uint8, device=device)

    for ksize in detect_kernels:
        kernel = torch.ones((1, 1, *ksize), dtype=torch.float32, device=device) / (ksize[0] * ksize[1])
        avg = torch.nn.functional.conv2d(batch, kernel, padding='same')[0, 0]
        mask_out += (avg > avg.mean() + sigma * avg.std()).to(torch.uint8)
        mask_out += (avg < avg.mean() - sigma * avg.std()).to(torch.uint8)
        grad_avg = torch.nn.functional.conv2d(
            grad_mag.unsqueeze(0).unsqueeze(0), kernel, padding='same')[0, 0]
        mask_out += (grad_avg < 1e-3).to(torch.uint8)

    mask_np  = (mask_out > 0).cpu().numpy().astype(np.uint8)
    mask_np  = remove_small_objects(mask_np)
    mask_out = torch.from_numpy(mask_np).to(device).float()

    for ksize in dilate_kernels:
        kernel  = torch.ones((1, 1, *ksize), dtype=torch.float32, device=device) / (ksize[0] * ksize[1])
        dilated = torch.nn.functional.conv2d(
            mask_out.unsqueeze(0).unsqueeze(0), kernel, padding='same')[0, 0]
        mask_out = ((mask_out + dilated) > 0.1).float()

    return (mask_out > 0).cpu().numpy().astype(np.uint8)


def fill_gaps(mrc_image_np, mask_np, tile_num=8):
    """Fill masked pixels with random local tissue samples."""
    h, w   = mrc_image_np.shape[1], mrc_image_np.shape[2]
    output = np.zeros_like(mrc_image_np, dtype=np.int8)

    for frame_i in range(mrc_image_np.shape[0]):
        frame     = mrc_image_np[frame_i]
        frame_out = frame.copy()
        for i in range(tile_num):
            for j in range(tile_num):
                th = h // tile_num
                tw = w // tile_num
                y0, y1 = i * th, (i + 1) * th
                x0, x1 = j * tw, (j + 1) * tw
                tile_mask = np.zeros((h, w), dtype=bool)
                tile_mask[y0:y1, x0:x1] = True
                gap_sel   = tile_mask & (mask_np == 1)
                good_sel  = tile_mask & (mask_np == 0)
                good_vals = frame[good_sel]
                n_gap     = int(np.sum(gap_sel))
                if n_gap > 0 and good_vals.size > 0:
                    frame_out[gap_sel] = good_vals[np.random.randint(0, good_vals.size, n_gap)]
                else:
                    frame_out[gap_sel] = 0
        output[frame_i] = frame_out

    return output


# Default kernel shapes used by build_gap_mask
DEFAULT_DETECT_KERNELS = [(301, 3), (3, 301)]
DEFAULT_DILATE_KERNELS = [(101, 3), (3, 101), (15, 15)]


def _parse_kernels(kernel_strings):
    """Parse a sequence of 'H,W' strings into a list of (H, W) tuples.

    Accepts either tuples/lists (already parsed) or strings like '301,3'.
    """
    result = []
    for k in kernel_strings:
        if isinstance(k, (list, tuple)):
            result.append(tuple(int(x) for x in k))
        else:
            h, w = k.split(',')
            result.append((int(h.strip()), int(w.strip())))
    return result


def _parse_index_list(entries):
    """Expand a mixed list of pixel indices and range strings into a flat int list.

    Each entry may be:
      - an integer             e.g. 3840
      - a string integer       e.g. "3840"
      - a closed range string  e.g. "3840-3860"  → 3840, 3841, …, 3860 (inclusive)

    Multiple entries and ranges are combined and deduplicated in sorted order.

    Examples
    --------
    >>> _parse_index_list([3840, 7680])
    [3840, 7680]
    >>> _parse_index_list(["3840-3850", 7680])
    [3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850, 7680]
    >>> _parse_index_list(["1-5", "8-10", 20])
    [1, 2, 3, 4, 5, 8, 9, 10, 20]
    """
    indices = set()
    for entry in entries:
        s = str(entry).strip()
        # Detect a range: contains '-' that is NOT a leading minus sign
        dash_pos = s.find('-', 1)   # search from position 1 to skip leading '-'
        if dash_pos != -1:
            start = int(s[:dash_pos].strip())
            end   = int(s[dash_pos + 1:].strip())
            if start > end:
                start, end = end, start  # tolerate reversed order
            indices.update(range(start, end + 1))
        else:
            indices.add(int(s))
    return sorted(indices)


def build_manual_mask(height, width, seam_rows, seam_cols, device,
                      dilate_kernels=None):
    """Build a binary gap mask from explicit row and column positions.

    Sets every pixel in the listed rows and columns to 1, then dilates the
    mask with ``dilate_kernels`` — the same dilation step used after
    auto-detection, so seam width is controlled by the same config values.

    Parameters
    ----------
    height, width : int
        Image dimensions in pixels.
    seam_rows : list[int]
        Row indices (y) where a seam runs horizontally across the full width.
    seam_cols : list[int]
        Column indices (x) where a seam runs vertically across the full height.
    device : torch.device
        Device used for dilation convolutions.
    dilate_kernels : list[(H, W)] or None
        Kernel shapes for dilation.  Defaults to DEFAULT_DILATE_KERNELS.
    """
    if dilate_kernels is None:
        dilate_kernels = DEFAULT_DILATE_KERNELS

    mask_np = np.zeros((height, width), dtype=np.uint8)
    for r in seam_rows:
        if 0 <= r < height:
            mask_np[r, :] = 1
    for c in seam_cols:
        if 0 <= c < width:
            mask_np[:, c] = 1

    # Dilate using the same torch convolution path as auto-detection
    mask_t = torch.from_numpy(mask_np).to(device).float()
    for ksize in dilate_kernels:
        kernel  = torch.ones((1, 1, *ksize), dtype=torch.float32, device=device) / (ksize[0] * ksize[1])
        dilated = torch.nn.functional.conv2d(
            mask_t.unsqueeze(0).unsqueeze(0), kernel, padding='same')[0, 0]
        mask_t  = ((mask_t + dilated) > 0.1).float()

    return (mask_t > 0).cpu().numpy().astype(np.uint8)


def process_image(image_path, output_dir, mask_dir, device,
                  skip_existing=True, sigma=5.0, tile_num=8,
                  detect_kernels=None, dilate_kernels=None,
                  seam_rows=None, seam_cols=None):
    """Detect (or use manual seams) and fill gaps for one blended frame MRC.

    If ``seam_rows`` or ``seam_cols`` is non-empty the manual mask path is
    used and GPU-based auto-detection is skipped entirely.
    """
    out_path  = os.path.join(output_dir, os.path.basename(image_path))
    mask_path = os.path.join(mask_dir,   os.path.basename(image_path))

    if skip_existing and os.path.exists(out_path):
        return

    with mrcfile.open(image_path, mode='r') as mrc:
        mrc_np = np.array(mrc.data)

    h, w = mrc_np.shape[-2], mrc_np.shape[-1]

    if seam_rows or seam_cols:
        # Manual mode — no detection needed, just build mask from given positions
        mask_np = build_manual_mask(
            h, w,
            seam_rows or [], seam_cols or [],
            device, dilate_kernels=dilate_kernels,
        )
    else:
        # Auto-detect mode
        mrc_tensor = torch.from_numpy(mrc_np).float().to(device)
        mrc_tensor = (mrc_tensor - mrc_tensor.mean()) / (mrc_tensor.std() + 1e-6)
        mask_np = build_gap_mask(mrc_tensor, device, sigma=sigma,
                                 detect_kernels=detect_kernels,
                                 dilate_kernels=dilate_kernels)

    write_mrc(mask_path, mask_np)
    output_np = fill_gaps(mrc_np, mask_np, tile_num=tile_num)
    write_mrc(out_path, output_np)
    print(f"  Filled: {os.path.basename(image_path)}")


def _process_with_gpu(args):
    (img_path, gpu_id, output_dir, mask_dir,
     skip_existing, sigma, tile_num,
     detect_kernels, dilate_kernels,
     seam_rows, seam_cols) = args
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    process_image(img_path, output_dir, mask_dir, device,
                  skip_existing, sigma, tile_num,
                  detect_kernels=detect_kernels,
                  dilate_kernels=dilate_kernels,
                  seam_rows=seam_rows,
                  seam_cols=seam_cols)


@click.command()
@click.option('--input-dir',  default='blended/frames',        show_default=True,
              help='Directory of blended frame MRC files.')
@click.option('--output-dir', default='blended/frames_filled',  show_default=True,
              help='Directory to write gap-filled MRC files.')
@click.option('--mask-dir',   default='blended/frames_masks',   show_default=True,
              help='Directory to write gap mask MRC files.')
@click.option('--gpus',       default='0', show_default=True,
              help='Comma-separated GPU IDs (e.g. "0,1,2") or "cpu".')
@click.option('--resume/--no-resume', default=True, show_default=True,
              help='Skip images already present in the output directory.')
@click.option('--sigma',      default=5.0, show_default=True,
              help='Standard-deviation multiplier for outlier gap detection.')
@click.option('--tile-num',   default=8,   show_default=True,
              help='Grid divisions per axis for local gap filling.')
@click.option('--detect-kernel', 'detect_kernel_strs',
              multiple=True, default=('301,3', '3,301'), show_default=True,
              help='Detection kernel as H,W (repeatable). '
                   'Ignored when --seam-row / --seam-col are supplied.')
@click.option('--dilate-kernel', 'dilate_kernel_strs',
              multiple=True, default=('101,3', '3,101', '15,15'), show_default=True,
              help='Dilation kernel as H,W (repeatable). '
                   'Applied in both auto-detect and manual-seam modes.')
@click.option('--seam-row', 'seam_row_strs', multiple=True, type=str, default=(),
              help='Horizontal seam row(s) to fill (repeatable). '
                   'Accepts a single pixel index (e.g. 3840) or an inclusive range '
                   '(e.g. 3840-3860). Providing any --seam-row or --seam-col '
                   'disables auto-detection.')
@click.option('--seam-col', 'seam_col_strs', multiple=True, type=str, default=(),
              help='Vertical seam column(s) to fill (repeatable). '
                   'Accepts a single pixel index or an inclusive range (e.g. 7680-7700).')
def main(input_dir, output_dir, mask_dir, gpus, resume, sigma, tile_num,
         detect_kernel_strs, dilate_kernel_strs, seam_row_strs, seam_col_strs):
    """Detect and fill blending-seam gaps in blended MRC frame stacks.

    \b
    Two modes:

      Auto-detect (default):
        python -m square_aperture_montage.remove_gaps --input-dir blended/frames

      Manual seams — single pixels or inclusive ranges, bypasses GPU detection:
        python -m square_aperture_montage.remove_gaps \\
          --seam-row 3840 --seam-row 7680 \\
          --seam-col 3840 --seam-col 7680 --gpus cpu

        python -m square_aperture_montage.remove_gaps \\
          --seam-row 3838-3842 --seam-row 7678-7682 \\
          --seam-col 3838-3842 --seam-col 7678-7682 --gpus cpu
    """
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    detect_kernels = _parse_kernels(detect_kernel_strs)
    dilate_kernels = _parse_kernels(dilate_kernel_strs)
    seam_rows      = _parse_index_list(seam_row_strs)
    seam_cols      = _parse_index_list(seam_col_strs)

    if seam_rows or seam_cols:
        print(f"Manual seam mode — rows: {seam_rows}  cols: {seam_cols}")
    else:
        print(f"Auto-detect mode — sigma: {sigma}  detect_kernels: {detect_kernels}")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir,   exist_ok=True)

    image_list = sorted(glob.glob(os.path.join(input_dir, '*.mrc')))
    if not image_list:
        print(f"No .mrc files found in '{input_dir}'. Exiting.")
        sys.exit(1)

    print(f"Found {len(image_list)} images to process.")

    if gpus.strip().lower() == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
        for img in tqdm.tqdm(image_list, desc="Filling gaps"):
            process_image(img, output_dir, mask_dir, device, resume, sigma, tile_num,
                          detect_kernels=detect_kernels, dilate_kernels=dilate_kernels,
                          seam_rows=list(seam_rows), seam_cols=list(seam_cols))
        return

    available = torch.cuda.device_count()
    gpu_ids   = [int(g.strip()) for g in gpus.split(',') if g.strip().isdigit()]
    gpu_ids   = [g for g in gpu_ids if g < available] or [0]

    print(f"Using GPUs: {gpu_ids}")
    task_args = [
        (img, gpu_ids[i % len(gpu_ids)], output_dir, mask_dir,
         resume, sigma, tile_num,
         detect_kernels, dilate_kernels,
         list(seam_rows), list(seam_cols))
        for i, img in enumerate(image_list)
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(_process_with_gpu, a) for a in task_args]
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures),
                                total=len(image_list), desc="Filling gaps"):
            try:
                future.result()
            except Exception as exc:
                print(f"  [ERROR] {exc}")


if __name__ == '__main__':
    main()
