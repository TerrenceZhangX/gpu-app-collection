/**
 * CUTLASS Volta TensorOp GEMM — Split-K + Horizontal Swizzle combo.
 *
 * Usage: ./gemm_volta_splitk_swizzle <M> <N> <K> [split_k]
 *
 * Combines two optimizations:
 *   1. GemmSplitKParallel — more CTAs via K-dimension splitting
 *   2. GemmHorizontalThreadblockSwizzle — L2-conscious CTA mapping
 *
 * Expected: the swizzle improves L2 reuse of B-matrix across split-K
 * CTA groups, potentially amplifying the split-K benefit for
 * memory-bound shapes.
 */

#include <iostream>
#include <cstdlib>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

// ---------------------------------------------------------------------------
// CUTLASS GEMM type definition (Split-K + Horizontal Swizzle)
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

using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;

// *** Horizontal swizzle for L2 reuse ***
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::GemmSplitKParallel<
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
    std::cerr << "Usage: " << argv[0] << " <M> <N> <K> [split_k] [gpu_id]" << std::endl;
    std::cerr << "  Split-K + Horizontal Swizzle GEMM" << std::endl;
    return 1;
  }

  int split_k = (argc >= 5) ? atoi(argv[4]) : 4;
  int gpu_id = (argc >= 6) ? atoi(argv[5]) : 0;
  CUDA_CHECK(cudaSetDevice(gpu_id));
  CUDA_CHECK(cudaFree(0));

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);

  double bytesA = (double)M * K * 2;
  double bytesB = (double)K * N * 2;
  double bytesD = (double)M * N * 4;
  double totalBytes = bytesA + bytesB + bytesD;
  double totalFLOPs = 2.0 * M * N * K;
  double ai = totalFLOPs / totalBytes;

  std::cout << "=== GEMM (Split-K=" << split_k << " + Horizontal Swizzle) ===" << std::endl;
  std::cout << "Shape: M=" << M << ", N=" << N << ", K=" << K << std::endl;
  std::cout << "Total memory: " << totalBytes / 1e6 << " MB" << std::endl;
  std::cout << "Arithmetic Intensity: " << ai << " FLOP/byte" << std::endl;

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

  typename Gemm::Arguments arguments{
      problem_size,
      tensor_a.device_ref(),
      tensor_b.device_ref(),
      tensor_c.device_ref(),
      tensor_d.device_ref(),
      {alpha, beta},
      split_k
  };

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(arguments));
  CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

  // Warmup
  CUTLASS_CHECK(gemm_op());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed run
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

  double timeCompute = totalFLOPs / 125e12;
  double timeMemory  = totalBytes / 897e9;
  std::cout << std::endl << "Roofline analysis:" << std::endl;
  std::cout << "  Min compute time: " << timeCompute * 1000 << " ms" << std::endl;
  std::cout << "  Min memory time:  " << timeMemory * 1000 << " ms" << std::endl;
  if (timeCompute > timeMemory) {
    std::cout << "  -> COMPUTE-BOUND (AI=" << ai << " > ridge point)" << std::endl;
  } else {
    std::cout << "  -> MEMORY-BOUND (AI=" << ai << " < ridge point)" << std::endl;
  }
  std::cout << std::endl;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
