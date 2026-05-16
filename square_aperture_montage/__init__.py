"""
square_aperture_montage
=======================
Python wrapper for IMOD that converts a 3×3 grid of cryo-EM tilt-series
(collected over a square aperture) into a single, large-field tilt-series
ready for downstream processing in IMOD, AreTomo, or similar.

Main scripts
------------
crop_images.py   — Remove blank borders from each tile image
blend_tiles.py   — Stitch tiles into one blended tilt-series per angle
remove_gaps.py   — Fill blending-seam artefacts with local texture

Utilities
---------
mdoc_reader.py   — Read, modify, and write SerialEM .mdoc metadata files
"""

from .mdoc_reader import (  # noqa: F401
    parse_mdoc_file,
    write_mdoc_file,
    get_z_section,
    get_all_z_values,
    get_tilt_angles,
    get_subframe_paths,
)

__version__ = "0.1.0"
__author__  = "Hamidreza Rahmani"
