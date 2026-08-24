"""Tests for crop centering in crop_images.detect_crop_boundaries.

These use synthetic numpy arrays only — no MRC files, no IMOD, no real data.
The invariant under test is that the returned window is *always* exactly
``crop_x × crop_y`` and is centered on the illuminated aperture, translating
(not truncating) when the centered window would spill past an image edge.

A small odd ``filter_size`` is used so the smoothed profile equals the raw
column/row sums, making the detected aperture extent crisp and deterministic.
"""

import inspect

import numpy as np
import pytest
from click.testing import CliRunner

from square_aperture_montage.crop_images import (
    detect_crop_boundaries,
    main as crop_main,
)


# Real detector geometry from the spec: ImageSize = 5760 4092 (width × height),
# leaving only 126 px of Y margin per side for a 3840 crop.
WIDTH, HEIGHT = 5760, 4092
CROP = 3840
FS = 1   # filter_size=1 → no smoothing → crisp, symmetric aperture edges


def make_field(cx, cy, ap_w=2000, ap_h=2000, width=WIDTH, height=HEIGHT):
    """A bright axis-aligned square of size ap_w × ap_h centered at (cx, cy)."""
    img = np.zeros((height, width), dtype=np.float32)
    x0, x1 = int(cx - ap_w / 2), int(cx + ap_w / 2)
    y0, y1 = int(cy - ap_h / 2), int(cy + ap_h / 2)
    img[y0:y1, x0:x1] = 1000.0
    return img


def _size(b):
    return (b.x_end - b.x_start, b.y_end - b.y_start)


def _win_center(b):
    return ((b.x_start + b.x_end) / 2, (b.y_start + b.y_end) / 2)


# ---------------------------------------------------------------------------
# 1. Centered aperture — window is exactly crop×crop and centered
# ---------------------------------------------------------------------------

def test_centered_aperture():
    img = make_field(WIDTH / 2, HEIGHT / 2, 3000, 3000)
    b = detect_crop_boundaries(img, filter_size=FS, crop_x=CROP, crop_y=CROP)

    assert _size(b) == (CROP, CROP)
    wx, wy = _win_center(b)
    assert abs(b.center_x - wx) <= 1
    assert abs(b.center_y - wy) <= 1
    assert b.shifted is False


# ---------------------------------------------------------------------------
# 2. Off-center aperture that still fits — the core regression
# ---------------------------------------------------------------------------

def test_off_center_fits():
    img = make_field(3200, 2100, 2000, 2000)
    b = detect_crop_boundaries(img, filter_size=FS, crop_x=CROP, crop_y=CROP)

    assert _size(b) == (CROP, CROP)
    wx, wy = _win_center(b)
    assert abs(b.center_x - wx) <= 1
    assert abs(b.center_y - wy) <= 1
    assert b.shifted is False


# ---------------------------------------------------------------------------
# 2b. Low-contrast aperture on a large constant background
# ---------------------------------------------------------------------------
# Motion-corrected averages carry a huge constant dose background; the
# illuminated aperture can be under 1% brighter than the dark border. A
# threshold relative to the profile's *absolute peak* is satisfied almost
# everywhere in that regime, so "detection" silently degenerates to the full
# frame and every crop centers on the sensor's geometric center instead of the
# true (off-center) aperture — a systematic shift that survives no matter how
# well the windowing/translation math (tested above) works.

def make_low_contrast_field(cx, cy, ap_w, ap_h, background=1e6, contrast=0.005,
                             width=WIDTH, height=HEIGHT):
    img = np.full((height, width), background, dtype=np.float64)
    x0, x1 = int(cx - ap_w / 2), int(cx + ap_w / 2)
    y0, y1 = int(cy - ap_h / 2), int(cy + ap_h / 2)
    img[y0:y1, x0:x1] = background * (1 + contrast)
    return img


def test_low_contrast_aperture_is_not_lost_to_full_frame():
    # Aperture offset from the geometric center by (+300, -200) px, only
    # 0.5% brighter than the background — comparable to real data.
    true_cx, true_cy = WIDTH / 2 + 300, HEIGHT / 2 - 200
    img = make_low_contrast_field(true_cx, true_cy, 3000, 3000)
    b = detect_crop_boundaries(img, filter_size=FS, crop_x=CROP, crop_y=CROP)

    assert _size(b) == (CROP, CROP)
    # Must land near the true (off-center) aperture, not the frame's
    # geometric center (WIDTH/2, HEIGHT/2) — the degenerate old behaviour.
    assert abs(b.center_x - true_cx) < 50
    assert abs(b.center_y - true_cy) < 50
    assert abs(b.center_x - WIDTH / 2) > 100
    assert abs(b.center_y - HEIGHT / 2) > 100


# ---------------------------------------------------------------------------
# 3. Aperture near the bottom edge — y_start would go negative (the real bug)
# ---------------------------------------------------------------------------

def test_bottom_edge_translates_not_truncates(capsys):
    # center_y = 1500 → ideal y_start = round(1500 - 1920) = -420 < 0
    img = make_field(WIDTH / 2, 1500, 1600, 1600)
    b = detect_crop_boundaries(img, filter_size=FS, crop_x=CROP, crop_y=CROP)

    assert _size(b) == (CROP, CROP)   # still exactly crop×crop, not undersized
    assert b.y_start == 0
    assert b.y_end == CROP
    assert b.shifted is True
    assert "[WARNING]" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 4. Aperture near the top / right edges — window pinned to far edge
# ---------------------------------------------------------------------------

def test_top_right_edges_translate(capsys):
    # center_x = 4000 → x_end would exceed width; center_y = 3000 → y_end > height
    img = make_field(4000, 3000, 1500, 1000)
    b = detect_crop_boundaries(img, filter_size=FS, crop_x=CROP, crop_y=CROP)

    assert _size(b) == (CROP, CROP)
    assert b.x_end == WIDTH
    assert b.y_end == HEIGHT
    assert b.x_start == WIDTH - CROP
    assert b.y_start == HEIGHT - CROP
    assert b.shifted is True
    assert "[WARNING]" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 5. Crop larger than the image — hard failure, not a clamp
# ---------------------------------------------------------------------------

def test_crop_larger_than_image_raises():
    img = make_field(WIDTH / 2, HEIGHT / 2, 2000, 2000)
    with pytest.raises(ValueError):
        detect_crop_boundaries(img, filter_size=FS, crop_x=CROP, crop_y=5000)


# ---------------------------------------------------------------------------
# 6. trim is gone — from the signature and from the CLI help
# ---------------------------------------------------------------------------

def test_trim_removed_from_signature():
    params = inspect.signature(detect_crop_boundaries).parameters
    assert "trim" not in params


def test_trim_removed_from_cli_help():
    res = CliRunner().invoke(crop_main, ["--help"])
    assert res.exit_code == 0
    assert "--trim" not in res.output
    assert "--crop-x" in res.output   # sanity: help really rendered


# ---------------------------------------------------------------------------
# 7. Output-size invariant across a grid of centers (incl. all four corners)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cx", [500, WIDTH // 2, WIDTH - 500])
@pytest.mark.parametrize("cy", [500, HEIGHT // 2, HEIGHT - 500])
def test_output_size_invariant(cx, cy):
    img = make_field(cx, cy, 800, 800)
    b = detect_crop_boundaries(img, filter_size=FS, crop_x=CROP, crop_y=CROP)

    assert _size(b) == (CROP, CROP)
    # Window always lands fully inside the image.
    assert 0 <= b.x_start and b.x_end <= WIDTH
    assert 0 <= b.y_start and b.y_end <= HEIGHT
