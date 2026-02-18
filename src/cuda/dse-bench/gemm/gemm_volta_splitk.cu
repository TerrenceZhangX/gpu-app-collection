/**
 * CUTLASS Volta TensorOp GEMM — Split-K Parallel variant.
 *
 * Optimization: Split the K-dimension across multiple CTA slices.
 * Each group of CTAs computes a partial sum over K/split_k elements,
 * then a reduction kernel sums the partials into the final output.
 *
 * This increases the total CTA count by split_k×, improving SM utilization
 * when the baseline has too few CTAs to fill all 80 SMs.
 *
 * Usage: ./gemm_volta_splitk <M> <N> <K> [split_k_slices] [gpu_id]
 *   Default split_k_slices = 4
 *
 * Baseline comparison:
 *   M=128, N=4096, K=4096: 32 CTAs → split_k=4 → 128 CTAs
 *   M=32,  N=11008, K=4096: 86 CTAs → split_k=4 → 344 CTAs
 */

#include <iostream>
#include <cstdlib>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

// ---------------------------------------------------------------------------
// Types — same as baseline
// ---------------------------------------------------------------------------

using ElementAccumulator = float;
using ElementComputeEpilogue = float;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = float;

using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm70;

// Same tile as baseline: 128×128×32
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;

// EpilogueOp for the final reduction (applied during reduction kernel)
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

constexpr int NumStages = 2;

using GemmSplitK = cutlass::gemm::device::GemmSplitKParallel<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    MMAOp, SmArch,
    ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
    EpilogueOp>;

// ---------------------------------------------------------------------------
// Helper macros
// ---------------------------------------------------------------------------

#define CUTLASS_CHECK(status)                                                  \
  do {                                                                         \
    cutlass::Status err = status;                                              \
    if (err != cutlass::Status::kSuccess) {                                    \
      std::cerr << "CUTLASS error: " << cutlassGetStatusString(err)            \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err)                   \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <M> <N> <K> [split_k_slices] [gpu_id]"
              << std::endl;
    return 1;
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);
  int split_k = (argc >= 5) ? atoi(argv[4]) : 4;
  int gpu_id = (argc >= 6) ? atoi(argv[5]) : 0;

  CUDA_CHECK(cudaSetDevice(gpu_id));
  CUDA_CHECK(cudaFree(0));

  double bytesA = (double)M * K * 2;
  double bytesB = (double)K * N * 2;
  double bytesD = (double)M * N * 4;
  double totalBytes = bytesA + bytesB + bytesD;
  double totalFLOPs = 2.0 * M * N * K;
  double ai = totalFLOPs / totalBytes;

  // CTA count calculation
  int cta_m = (M + 127) / 128;
  int cta_n = (N + 127) / 128;
  int base_ctas = cta_m * cta_n;

  std::cout << "=== GEMM Split-K Parallel (CUTLASS Volta TensorOp FP16) ===" << std::endl;
  std::cout << "Shape: M=" << M << ", N=" << N << ", K=" << K << std::endl;
  std::cout << "Split-K slices: " << split_k << std::endl;
  std::cout << "Tile: 128x128x32, Warp: 64x64x32, MMA: 8x8x4" << std::endl;
  std::cout << "Base CTAs: " << base_ctas << " → with split-K: "
            << base_ctas * split_k << std::endl;
  std::cout << "K per slice: " << K / split_k << std::endl;
  std::cout << "Arithmetic Intensity: " << ai << " FLOP/byte" << std::endl;

  // ---- Allocate & init tensors ----
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.mk());
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.kn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(problem_size.mn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(problem_size.mn());

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(), 1, ElementInputA(4), ElementInputA(-4), 0);
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(), 1, ElementInputB(4), ElementInputB(-4), 0);
  cutlass::reference::host::TensorFill(tensor_c.host_view());
  cutlass::reference::host::TensorFill(tensor_d.host_view());

  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();

  ElementComputeEpilogue alpha = 1.0f;
  ElementComputeEpilogue beta = 0.0f;

  typename GemmSplitK::Arguments arguments{
      problem_size,
      tensor_a.device_ref(),
      tensor_b.device_ref(),
      tensor_c.device_ref(),
      tensor_d.device_ref(),
      {alpha, beta},
      split_k
  };

  size_t workspace_size = GemmSplitK::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  std::cout << "Workspace: " << workspace_size / 1024.0 << " KB" << std::endl;

  GemmSplitK gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

  // ---- Warmup ----
  CUTLASS_CHECK(gemm_op());
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- Timed run ----
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  CUTLASS_CHECK(gemm_op());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  double seconds = ms / 1000.0;
  double tflops = totalFLOPs / seconds / 1e12;
  double gbps = totalBytes / seconds / 1e9;

  std::cout << "Kernel time: " << ms << " ms" << std::endl;
  std::cout << "TFLOPS:      " << tflops << std::endl;
  std::cout << "Effective BW: " << gbps << " GB/s" << std::endl;
  std::cout << "V100 TC FP16 utilization: " << tflops / 125.0 * 100.0 << "%" << std::endl;
  std::cout << "V100 HBM BW utilization:  " << gbps / 897.0 * 100.0 << "%" << std::endl;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
