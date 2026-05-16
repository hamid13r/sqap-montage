#!/usr/bin/env bash
# =============================================================================
# example_workflow.sh — End-to-end square-aperture montage pipeline
#
# Run from your DATA directory (the folder containing mdocs/ and frames/).
#
# Usage:
#   cd /data1/users/Krios_Data/HRR/HRR036_1_TEM_250220
#   bash /path/to/sqap-montage/examples/example_workflow.sh
#
# Override any default with an environment variable:
#   BLEND_SIZE=9000 CROP_X=3200 bash example_workflow.sh
# =============================================================================

set -euo pipefail

MDOC_DIR="${MDOC_DIR:-mdocs}"
AVERAGES_DIR="${AVERAGES_DIR:-frames/averages}"
FRAMES_DIR="${FRAMES_DIR:-frames}"
CROP_X="${CROP_X:-3840}"
CROP_Y="${CROP_Y:-3840}"
BLEND_SIZE="${BLEND_SIZE:-11664}"
NUM_FRAMES="${NUM_FRAMES:-4}"
GPUS="${GPUS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-blended}"
PROCESSING_DIR="${PROCESSING_DIR:-processing}"
TS_FILTER="${TS_FILTER:-}"   # leave empty to process all series

echo "============================================================"
echo "  Square-aperture montage pipeline"
echo "  Working directory: $(pwd)"
echo "============================================================"

echo ""
echo "[1/3] Cropping tile borders..."
sam-crop \
    --input-dir   "$AVERAGES_DIR" \
    --output-dir  cropped \
    --frames-dir  "$FRAMES_DIR" \
    --crop-x      "$CROP_X" \
    --crop-y      "$CROP_Y"

echo ""
echo "[2/3] Blending tiles..."
TS_FLAGS=""
for ts in $TS_FILTER; do TS_FLAGS="$TS_FLAGS --ts $ts"; done

sam-blend \
    --mdoc-dir       "$MDOC_DIR" \
    --averages-dir   cropped/averages \
    --frames-dir     cropped/frames \
    --output-dir     "$OUTPUT_DIR" \
    --processing-dir "$PROCESSING_DIR" \
    --blend-size     "$BLEND_SIZE" \
    --num-frames     "$NUM_FRAMES" \
    $TS_FLAGS

echo ""
echo "[3/3] Filling blending-seam gaps..."
sam-fill \
    --input-dir  "$OUTPUT_DIR/frames" \
    --output-dir "$OUTPUT_DIR/frames_filled" \
    --mask-dir   "$OUTPUT_DIR/frames_masks" \
    --gpus       "$GPUS" \
    --resume

echo ""
echo "============================================================"
echo "  Done!  Results in: $OUTPUT_DIR/"
echo "  MDOCs: $OUTPUT_DIR/averages/mdocs/  and  $OUTPUT_DIR/frames/mdocs/"
echo "============================================================"
