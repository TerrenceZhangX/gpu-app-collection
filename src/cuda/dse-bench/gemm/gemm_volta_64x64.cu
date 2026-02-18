/**
 * CUTLASS Volta TensorOp GEMM — Small tile (64×64×32) variant.
 *
 * Optimization: Use a smaller threadblock tile (64×64×32 vs baseline 128×128×32).
 * This generates 4× more CTAs for the same problem, dramatically improving SM
 * utilization when the problem doesn't produce enough CTAs to fill all 80 SMs.
 *
 * Trade-offs:
 * - 4× more CTAs → better SM occupancy for small M
 * - Each CTA computes 4× fewer output elements → reduced data reuse per CTA
 * - 4 warps × 32 threads = 128 threads/CTA (same)
 * - Lower register pressure → potentially higher occupancy per SM
 *
 * CTA counts for our shapes:
 *   Baseline 128×128:  M=8→32, M=32→86, M=128→32, M=4096→2752
 *   This 64×64:        M=8→64, M=32→172, M=128→128, M=4096→11008
 *
 * Usage: ./gemm_volta_64x64 <M> <N> <K> [gpu_id]
 */

#include <iostream>
#include <cstdlib>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
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

// *** KEY CHANGE: 64×64×32 tile (vs 128×128×32 baseline) ***
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<32, 32, 32>;
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementComputeEpilogue>;

constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    MMAOp, SmArch,
    ShapeMMAThreadBlock, ShapeMMAWarp, ShapeMMAOp,
    EpilogueOp, SwizzleThreadBlock, NumStages>;

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
    std::cerr << "Usage: " << argv[0] << " <M> <N> <K> [gpu_id]" << std::endl;
    return 1;
  }

  int gpu_id = (argc >= 5) ? atoi(argv[4]) : 0;
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

  int cta_m = (M + 63) / 64;
  int cta_n = (N + 63) / 64;
  int total_ctas = cta_m * cta_n;

  std::cout << "=== GEMM 64x64 Tile (CUTLASS Volta TensorOp FP16) ===" << std::endl;
  std::cout << "Shape: M=" << M << ", N=" << N << ", K=" << K << std::endl;
  std::cout << "Tile: 64x64x32, Warp: 32x32x32, MMA: 8x8x4" << std::endl;
  std::cout << "CTA grid: " << cta_m << " x " << cta_n
            << " = " << total_ctas << " CTAs" << std::endl;
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
      1
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

  // Roofline
  double timeCompute = totalFLOPs / 125e12;
  double timeMemory = totalBytes / 897e9;
  std::cout << std::endl << "Roofline:" << std::endl;
  if (timeCompute > timeMemory) {
    std::cout << "  -> COMPUTE-BOUND (AI=" << ai << ")" << std::endl;
  } else {
    std::cout << "  -> MEMORY-BOUND (AI=" << ai << ")" << std::endl;
  }
  std::cout << std::endl;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaDeviceReset());
  return 0;
}
