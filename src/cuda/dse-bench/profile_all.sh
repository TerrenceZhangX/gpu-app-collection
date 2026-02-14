#!/bin/bash
# Run all DSE bench profiling
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Building all benchmarks..."
make -C "$SCRIPT_DIR" -j

echo ""
echo "========================================"
echo " Part 1: vectorAdd sweep"
echo "========================================"
bash "$SCRIPT_DIR/profile_vecadd.sh"

echo ""
echo "========================================"
echo " Part 2: GEMM sweep"  
echo "========================================"
bash "$SCRIPT_DIR/profile_gemm.sh"

echo ""
echo "All results saved to:"
echo "  $SCRIPT_DIR/results_vecadd.csv"
echo "  $SCRIPT_DIR/results_gemm.csv"
