#!/bin/bash
# Profile GEMM (CUTLASS Volta TensorOp FP16) with LLM Transformer shapes on V100
# Reference: llmcompass/software_model/transformer.py
# Model config: LLaMA-7B style (d=4096, ffn=11008, d_h=128, single GPU no TP)
#
# Prefill phase: M = seq_len (compute-bound for large seq)
# Decode phase:  M = batch_size (memory-bound, weight-loading dominated)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$SCRIPT_DIR/gemm/gemm_volta"

if [ ! -f "$BIN" ]; then
  echo "Building gemm_volta..."
  make -C "$SCRIPT_DIR" gemm
fi

echo "================================================================"
echo "LLM Transformer GEMM sweep - V100 (125 TFLOPS FP16 TC, 897 GB/s HBM)"
echo "Model: LLaMA-7B style (d=4096, ffn=11008)"
echo "================================================================"
echo ""

# Format: "label M N K"
# QKV/O projection: [M, 4096] × [4096, 4096]
# FFN up:           [M, 4096] × [4096, 11008]
# FFN down:         [M, 11008] × [11008, 4096]
SHAPES=(
  # === Decode phase (M = batch_size, memory-bound) ===
  "decode-QKV-bs1       8 4096 4096"
  "decode-FFN_up-bs1    8 4096 11008"
  "decode-FFN_down-bs1  8 11008 4096"
  "decode-QKV-bs32      32 4096 4096"
  "decode-FFN_up-bs32   32 4096 11008"
  "decode-FFN_down-bs32 32 11008 4096"
  "decode-QKV-bs128     128 4096 4096"
  "decode-FFN_up-bs128  128 4096 11008"
  "decode-FFN_down-bs128 128 11008 4096"
  # === Prefill phase (M = seq_len, compute-bound) ===
  "prefill-QKV-s512     512 4096 4096"
  "prefill-FFN_up-s512  512 4096 11008"
  "prefill-FFN_down-s512 512 11008 4096"
  "prefill-QKV-s2048    2048 4096 4096"
  "prefill-FFN_up-s2048 2048 4096 11008"
  "prefill-FFN_down-s2048 2048 11008 4096"
  "prefill-QKV-s4096    4096 4096 4096"
  "prefill-FFN_up-s4096 4096 4096 11008"
  "prefill-FFN_down-s4096 4096 11008 4096"
)

echo "label,M,N,K,AI,kernel_ms,tflops,eff_bw_gbps,tc_util_pct,bw_util_pct,bottleneck" > "$SCRIPT_DIR/results_gemm.csv"

MAX_RETRIES=3

for SHAPE in "${SHAPES[@]}"; do
  read -r LABEL M N K <<< "$SHAPE"
  echo "--- $LABEL: M=$M N=$N K=$K ---"

  SUCCESS=0
  for ATTEMPT in $(seq 1 $MAX_RETRIES); do
    "$BIN" "$M" "$N" "$K" 2>&1 | tee /tmp/_gemm_out.txt
    if grep -q "Kernel time:" /tmp/_gemm_out.txt; then
      SUCCESS=1
      break
    fi
    echo "  [retry $ATTEMPT/$MAX_RETRIES] CUDA init failed, waiting 1s..."
    sleep 1
  done

  if [ $SUCCESS -eq 1 ]; then
    AI=$(grep "Arithmetic Intensity:" /tmp/_gemm_out.txt | awk '{print $3}')
    KERNEL_MS=$(grep "Kernel time:" /tmp/_gemm_out.txt | awk '{print $3}')
    TFLOPS=$(grep "TFLOPS:" /tmp/_gemm_out.txt | awk '{print $2}')
    EFF_BW=$(grep "Effective BW:" /tmp/_gemm_out.txt | awk '{print $3}')
    TC_UTIL=$(grep "TC FP16 utilization:" /tmp/_gemm_out.txt | awk '{print $5}' | tr -d '%')
    BW_UTIL=$(grep "HBM BW utilization:" /tmp/_gemm_out.txt | awk '{print $5}' | tr -d '%')
    BOTTLENECK=$(grep -o "COMPUTE-BOUND\|MEMORY-BOUND" /tmp/_gemm_out.txt || echo "UNKNOWN")
  else
    echo "  [FAILED] All $MAX_RETRIES attempts failed for $LABEL"
    AI=""
    KERNEL_MS=""
    TFLOPS=""
    EFF_BW=""
    TC_UTIL=""
    BW_UTIL=""
    BOTTLENECK="FAILED"
  fi

  echo "$LABEL,$M,$N,$K,$AI,$KERNEL_MS,$TFLOPS,$EFF_BW,$TC_UTIL,$BW_UTIL,$BOTTLENECK" >> "$SCRIPT_DIR/results_gemm.csv"
  echo ""
done

echo "================================================================"
echo "Results saved to $SCRIPT_DIR/results_gemm.csv"
echo "================================================================"
