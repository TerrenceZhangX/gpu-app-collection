/**
 * vectorAdd_unroll4: C[i] = A[i] + B[i] with grid-stride loop + 4× unrolling.
 *
 * Optimization: Each thread processes ~4 elements using grid-stride pattern.
 * Grid-stride ensures adjacent threads ALWAYS access adjacent memory (coalesced).
 *
 * Expected speedup sources:
 *   - 4× fewer serial R2 dependency episodes on critical path
 *   - Increased MLP — multiple independent loads in flight per warp
 *   - Better scheduler utilization
 *
 * Usage: ./vectorAdd_unroll4 <numElements>
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void vectorAdd_unroll4(const float *A, const float *B, float *C,
                                   long long numElements) {
  long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
  long long gridStride = (long long)blockDim.x * gridDim.x;

  // Grid-stride loop: adjacent threads access adjacent memory (coalesced).
  // #pragma unroll 4 tells NVCC to unroll 4 iterations, issuing 4 loads
  // before first store → breaks R2 serial dependency, increases MLP.
  #pragma unroll 4
  for (long long i = tid; i < numElements; i += gridStride) {
    C[i] = A[i] + B[i];
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <numElements> [gpu_id]\n", argv[0]);
    return 1;
  }

  int gpu_id = (argc >= 3) ? atoi(argv[2]) : 0;
  CHECK_CUDA(cudaSetDevice(gpu_id));
  CHECK_CUDA(cudaFree(0));

  long long numElements = atoll(argv[1]);
  size_t size = numElements * sizeof(float);
  double totalBytes = 3.0 * size;
  double totalFLOPs = (double)numElements;

  printf("=== vectorAdd_unroll4 (grid-stride + 4x unroll, scalar loads) ===\n");
  printf("Elements:  %lld\n", numElements);
  printf("Data size: %.2f MB total traffic\n", totalBytes / (1024.0 * 1024.0));

  // Host alloc & init
  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);
  if (!h_A || !h_B || !h_C) {
    fprintf(stderr, "Host malloc failed\n");
    return 1;
  }
  for (long long i = 0; i < numElements; i++) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  // Device alloc
  float *d_A, *d_B, *d_C;
  CHECK_CUDA(cudaMalloc(&d_A, size));
  CHECK_CUDA(cudaMalloc(&d_B, size));
  CHECK_CUDA(cudaMalloc(&d_C, size));

  CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

  // Launch 1/4 of baseline grid — each thread processes ~4 elements
  int threadsPerBlock = 256;
  long long totalThreads = (numElements + 3) / 4;
  int blocksPerGrid = (int)((totalThreads + threadsPerBlock - 1) / threadsPerBlock);
  printf("Launch: %d blocks x %d threads (~4 elements/thread via grid-stride)\n",
         blocksPerGrid, threadsPerBlock);

  // Warmup
  vectorAdd_unroll4<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timed run
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  vectorAdd_unroll4<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  double seconds = ms / 1000.0;
  double gbps = totalBytes / seconds / 1e9;
  double gflops = totalFLOPs / seconds / 1e9;

  printf("Kernel time: %.4f ms\n", ms);
  printf("Effective BW: %.2f GB/s\n", gbps);
  printf("GFLOPS:       %.2f\n", gflops);
  printf("V100 HBM BW utilization: %.1f%%\n", gbps / 897.0 * 100.0);

  // Verify
  CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
  for (int i = 0; i < 10 && i < numElements; i++) {
    if (fabsf(h_A[i] + h_B[i] - h_C[i]) > 1e-5f) {
      fprintf(stderr, "FAILED at element %d\n", i);
      return 1;
    }
  }
  printf("Verification: PASSED\n\n");

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));
  free(h_A); free(h_B); free(h_C);

  CHECK_CUDA(cudaDeviceReset());
  return 0;
}
