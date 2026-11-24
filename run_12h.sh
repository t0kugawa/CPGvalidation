#!/usr/bin/env bash
# Production 12-hour CPG + MuJoCo + PERSONA run
# Usage:
#   ./run_12h.sh
# or in the background:
#   nohup ./run_12h.sh > runs/run_12h.log 2>&1 &

set -e

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SOURCE_DIR"

# Activate conda environment if available
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate CPGV
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate CPGV
fi

# Set MuJoCo headless rendering with EGL / OSMESA
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="runs/prod_12h_${TIMESTAMP}"
mkdir -p "$OUT_DIR"

echo "=========================================================="
echo "Starting 12-hour production run at $(date)"
echo "Output directory: $OUT_DIR"
echo "MuJoCo GL: $MUJOCO_GL"
echo "=========================================================="

python run.py \
    --preset full \
    --sim-time 43200 \
    --budget-s 43200 \
    --render \
    --belt model \
    --game selected \
    --lr-start 0.03 \
    --snapshot-s 900 \
    --out "$OUT_DIR"

echo "=========================================================="
echo "Production run completed at $(date)"
echo "Output files in $OUT_DIR:"
ls -lh "$OUT_DIR"
echo "=========================================================="
