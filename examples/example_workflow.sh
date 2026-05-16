#!/usr/bin/env bash
# =============================================================================
# example_workflow.sh — End-to-end square-aperture montage pipeline
#
# Run from your DATA directory (the folder containing mdocs/ and frames/),
# pointing at a pipeline.yaml config file.
#
# Usage:
#   cd /data1/users/Krios_Data/HRR/HRR036_1_TEM_250220
#   bash /path/to/sqap-montage/examples/example_workflow.sh
#
# The script looks for pipeline.yaml in the current directory by default.
# Override with: CONFIG=/path/to/pipeline.yaml bash example_workflow.sh
# =============================================================================

set -euo pipefail

CONFIG="${CONFIG:-pipeline.yaml}"

# Resolve the sqap_montage.py path relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQAP="${SQAP:-python ${SCRIPT_DIR}/../sqap_montage.py}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file '$CONFIG' not found."
    echo "Generate one with: $SQAP write-config pipeline.yaml"
    exit 1
fi

echo "============================================================"
echo "  Square-aperture montage pipeline"
echo "  Config:           $CONFIG"
echo "  Working directory: $(pwd)"
echo "============================================================"

echo ""
echo "[1/4] Cropping tile borders..."
$SQAP crop --config "$CONFIG"

echo ""
echo "[2/4] Blending tiles..."
$SQAP blend --config "$CONFIG"

echo ""
echo "[3/4] Filling blending-seam gaps..."
$SQAP fill --config "$CONFIG"

echo ""
echo "[4/4] Building blended mdoc files..."
$SQAP make-mdoc --config "$CONFIG"

echo ""
echo "============================================================"
echo "  Done!"
echo "  Blended tilt-series:  blended/averages/  and  blended/frames/"
echo "  Blended mdocs:        blended_mdocs/"
echo "============================================================"
