# DSE Bench — LLM Inference GPU Micro-Benchmarks

Canonical GPU kernels for Design Space Exploration (DSE) of LLM inference workloads.
Shapes are derived from LLaMA-7B style Transformer blocks (`d_model=4096`, `ffn_dim=11008`),
referencing `llmcompass/software_model/transformer.py`.

These benchmarks are designed to feed into AccelSim for microarchitectural bottleneck analysis.

## Kernels

| Kernel | LLM Role | AI (FLOP/Byte) | Bottleneck |
|--------|----------|-----------------|-----------|
| **vectorAdd** | Residual add (`x + attn(x)`, `x + ffn(x)`) | 0.083 (fixed) | Always memory-bound |
| **gemm_volta** | QKV/O projection, FFN up/down | Varies with M/N/K | Decode: memory-bound; Prefill: compute-bound |

## Build & Run

```bash
# Build (inside Docker container)
make all

# Profile all
./profile_vecadd.sh   # Residual add sweep
./profile_gemm.sh     # GEMM sweep
```

## V100 Reference Numbers

| Metric | Value |
|--------|-------|
| Peak HBM BW | 897 GB/s |
| FP16 Tensor Core | 125 TFLOPS |
| FP32 CUDA Core | 15.7 TFLOPS |
| L2 Cache | 6 MB |
| Ridge Point (TC) | ~139 FLOP/byte |

---

## Profiling Results (V100-PCIE-32GB)

### Residual Add (vectorAdd)

Shape: `[batch, seq_len, d_model]` → `num_elements = batch × seq_len × 4096`

AI = 0.083 FLOP/byte (always memory-bound). Bottleneck transitions from latency-bound to BW-bound as data exceeds L2 cache (6 MB).

| Label | Elements | Size (MB) | Kernel (ms) | Eff BW (GB/s) | BW Util |
|-------|----------|-----------|-------------|---------------|---------|
| decode-bs1 | 4,096 | 0.016 | 0.009 | 5.3 | 0.6% |
| decode-bs8 | 32,768 | 0.125 | 0.008 | 48.0 | 5.4% |
| decode-bs32 | 131,072 | 0.5 | 0.009 | 170.7 | 19.0% |
| decode-bs64 | 262,144 | 1.0 | 0.009 | 341.3 | 38.1% |
| decode-bs128 | 524,288 | 2.0 | 0.011 | 558.6 | 62.3% |
| decode-bs256 | 1,048,576 | 4.0 | 0.023 | 558.6 | 62.3% |
| prefill-s128 | 524,288 | 2.0 | 0.011 | 558.6 | 62.3% |
| prefill-s512 | 2,097,152 | 8.0 | 0.038 | 664.2 | 74.0% |
| prefill-s1024 | 4,194,304 | 16.0 | 0.070 | 722.8 | 80.6% |
| prefill-s2048 | 8,388,608 | 32.0 | 0.130 | 774.2 | 86.3% |
| prefill-s4096 | 16,777,216 | 64.0 | 0.255 | 789.6 | 88.0% |

### GEMM (CUTLASS Volta TensorOp FP16)

Shapes: QKV/O projection `[M, 4096] × [4096, 4096]`, FFN up `[M, 4096] × [4096, 11008]`, FFN down `[M, 11008] × [11008, 4096]`.

Decode (M = batch_size) is memory-bound; Prefill (M = seq_len) is compute-bound.

| Label | M | N | K | AI | Kernel (ms) | TFLOPS | BW (GB/s) | TC Util | BW Util | Bottleneck |
|-------|---|---|---|----|-------------|--------|-----------|---------|---------|------------|
| decode-QKV-bs1 | 8 | 4096 | 4096 | 8.0 | 0.139 | 1.9 | 242 | 1.5% | 27.0% | MEM |
| decode-FFN_up-bs1 | 8 | 4096 | 11008 | 8.0 | 0.349 | 2.1 | 259 | 1.7% | 28.9% | MEM |
| decode-FFN_down-bs1 | 8 | 11008 | 4096 | 8.0 | 0.240 | 3.0 | 378 | 2.4% | 42.2% | MEM |
| decode-QKV-bs32 | 32 | 4096 | 4096 | 31.3 | 0.140 | 7.7 | 245 | 6.1% | 27.3% | MEM |
| decode-FFN_up-bs32 | 32 | 4096 | 11008 | 31.6 | 0.353 | 8.2 | 259 | 6.5% | 28.8% | MEM |
| decode-FFN_down-bs32 | 32 | 11008 | 4096 | 31.4 | 0.240 | 12.0 | 383 | 9.6% | 42.7% | MEM |
| decode-QKV-bs128 | 128 | 4096 | 4096 | 117 | 0.155 | 27.8 | 237 | 22.2% | 26.5% | MEM |
| decode-FFN_up-bs128 | 128 | 4096 | 11008 | 121 | 0.389 | 29.7 | 244 | 23.7% | 27.2% | MEM |
| decode-FFN_down-bs128 | 128 | 11008 | 4096 | 119 | 0.246 | 47.0 | 394 | 37.6% | 43.9% | MEM |
| prefill-QKV-s512 | 512 | 4096 | 4096 | 372 | 0.243 | 70.8 | 190 | 56.6% | 21.2% | COMP |
| prefill-FFN_up-s512 | 512 | 4096 | 11008 | 420 | 0.621 | 74.4 | 177 | 59.5% | 19.7% | COMP |
| prefill-FFN_down-s512 | 512 | 11008 | 4096 | 395 | 0.626 | 73.8 | 187 | 59.0% | 20.8% | COMP |
| prefill-QKV-s2048 | 2048 | 4096 | 4096 | 819 | 0.853 | 80.6 | 98 | 64.4% | 11.0% | COMP |
| prefill-FFN_up-s2048 | 2048 | 4096 | 11008 | 1094 | 2.214 | 83.4 | 76 | 66.7% | 8.5% | COMP |
| prefill-FFN_down-s2048 | 2048 | 11008 | 4096 | 937 | 2.120 | 87.1 | 93 | 69.7% | 10.4% | COMP |
| prefill-QKV-s4096 | 4096 | 4096 | 4096 | 1024 | 1.568 | 87.7 | 86 | 70.1% | 9.5% | COMP |
| prefill-FFN_up-s4096 | 4096 | 4096 | 11008 | 1493 | 4.295 | 86.0 | 58 | 68.8% | 6.4% | COMP |
| prefill-FFN_down-s4096 | 4096 | 11008 | 4096 | 1215 | 4.178 | 88.4 | 73 | 70.7% | 8.1% | COMP |

---

## Selected Shapes for AccelSim DSE

Representative subset covering all bottleneck regimes:

### Residual Add (vectorAdd)

| Config | Elements | Regime | Rationale |
|--------|----------|--------|-----------|
| `vectorAdd 4096` | 4K | **Latency-bound** | Decode bs=1, data fits in L1, BW util < 1% |
| `vectorAdd 262144` | 256K | **Transition** | Decode bs=64, data ~1MB (L2 resident), BW util ~38% |
| `vectorAdd 16777216` | 16M | **BW-bound** | Prefill s=4096, data ~64MB (HBM streaming), BW util ~88% |

### GEMM (gemm_volta)

| Config | M×N×K | Regime | Rationale |
|--------|-------|--------|-----------|
| `gemm_volta 8 4096 4096` | 8×4096×4096 | **Latency-bound** | Decode QKV bs=1, AI=8, BW util 27%, too few warps to saturate HBM |
| `gemm_volta 32 11008 4096` | 32×11008×4096 | **BW-bound** | Decode FFN down bs=32, AI=31, BW util 43%, weight-loading dominated |
| `gemm_volta 128 4096 4096` | 128×4096×4096 | **Ridge (mem→comp)** | Decode QKV bs=128, AI=117 (approaching ridge point 139), transition zone |
| `gemm_volta 4096 11008 4096` | 4096×11008×4096 | **Compute-bound** | Prefill FFN down s=4096, AI=1215, TC util ~71% |
