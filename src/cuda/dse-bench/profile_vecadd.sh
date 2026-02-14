#!/bin/bash
# Profile vectorAdd as LLM Transformer residual-add on V100
# Reference: Transformer block has 2 residual adds per layer:
#   1. x_out = x + attention(x)   shape [b, seq_len, d]
#   2. x_out = x + ffn(x)         shape [b, seq_len, d]
# Model: LLaMA-7B style (d=4096)
# Elements = batch × seq_len × 4096
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$SCRIPT_DIR/vectorAdd/vectorAdd"

if [ ! -f "$BIN" ]; then
  echo "Building vectorAdd..."
  make -C "$SCRIPT_DIR" vecadd
fi

echo "================================================================"
echo "Residual Add sweep (vectorAdd) - V100 (897 GB/s HBM, 6 MB L2)"
echo "Model: LLaMA-7B (d=4096), AI=0.083 FLOP/byte (always memory-bound)"
echo "================================================================"
echo ""

# Format: "label num_elements"
# Decode: elements = bs × 1 × 4096
# Prefill: elements = 1 × seq_len × 4096
CONFIGS=(
  "decode-bs1     4096"
  "decode-bs8     32768"
  "decode-bs32    131072"
  "decode-bs64    262144"
  "decode-bs128   524288"
  "decode-bs256   1048576"
  "prefill-s128   524288"
  "prefill-s512   2097152"
  "prefill-s1024  4194304"
  "prefill-s2048  8388608"
  "prefill-s4096  16777216"
)

echo "label,size_elements,size_MB,kernel_ms,eff_bw_gbps,bw_util_pct" > "$SCRIPT_DIR/results_vecadd.csv"

MAX_RETRIES=3

for CFG in "${CONFIGS[@]}"; do
  read -r LABEL N <<< "$CFG"
  echo "--- $LABEL: N=$N ---"

  SUCCESS=0
  for ATTEMPT in $(seq 1 $MAX_RETRIES); do
    "$BIN" "$N" 2>&1 | tee /tmp/_vecadd_out.txt
    if grep -q "Kernel time:" /tmp/_vecadd_out.txt; then
      SUCCESS=1
      break
    fi
    echo "  [retry $ATTEMPT/$MAX_RETRIES] CUDA init failed, waiting 1s..."
    sleep 1
  done

  if [ $SUCCESS -eq 1 ]; then
    KERNEL_MS=$(grep "Kernel time:" /tmp/_vecadd_out.txt | awk '{print $3}')
    EFF_BW=$(grep "Effective BW:" /tmp/_vecadd_out.txt | awk '{print $3}')
    BW_UTIL=$(grep "HBM BW utilization:" /tmp/_vecadd_out.txt | awk '{print $5}' | tr -d '%')
  else
    echo "  [FAILED] All $MAX_RETRIES attempts failed for $LABEL"
    KERNEL_MS=""
    EFF_BW=""
    BW_UTIL=""
  fi

  SIZE_MB=$(awk "BEGIN {printf \"%.4f\", $N * 4.0 / 1048576}")
  echo "$LABEL,$N,$SIZE_MB,$KERNEL_MS,$EFF_BW,$BW_UTIL" >> "$SCRIPT_DIR/results_vecadd.csv"
  echo ""
done

echo "================================================================"
echo "Results saved to $SCRIPT_DIR/results_vecadd.csv"
echo "================================================================"
