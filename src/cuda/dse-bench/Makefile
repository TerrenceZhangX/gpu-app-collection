CUDA_HOME    ?= /usr/local/cuda
NVCC         := $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS   := -O3 -arch=sm_70 -std=c++17

# CUTLASS headers (relative to gpu-app-collection/src/cuda/)
CUTLASS_DIR  ?= $(shell cd .. && pwd)/cutlass-bench
CUTLASS_INC  := -I$(CUTLASS_DIR)/include -I$(CUTLASS_DIR)/tools/util/include \
                -I$(CUTLASS_DIR)/examples/common
CUTLASS_DEFS := -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1

.PHONY: all vecadd gemm clean

all: vecadd gemm

# ── vectorAdd ────────────────────────────────────────────────────
vecadd: vectorAdd/vectorAdd
vectorAdd/vectorAdd: vectorAdd/vectorAdd.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# ── GEMM (CUTLASS Volta TensorOp) ───────────────────────────────
gemm: gemm/gemm_volta
gemm/gemm_volta: gemm/gemm_volta.cu
	$(NVCC) $(NVCC_FLAGS) $(CUTLASS_INC) $(CUTLASS_DEFS) --expt-relaxed-constexpr -o $@ $<

clean:
	rm -f vectorAdd/vectorAdd gemm/gemm_volta
