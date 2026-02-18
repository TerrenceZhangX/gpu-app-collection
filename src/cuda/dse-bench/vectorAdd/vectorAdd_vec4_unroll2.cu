/**
 * vectorAdd_vec4_unroll2: C[i] = A[i] + B[i] with float4 + grid-stride 2× unroll.
 *
 * Optimization: Combined LDG.128 vectorization + grid-stride loop with 2× unroll.
 * Each thread processes ~8 elements (2 float4 iterations via grid stride).
 *
 * Grid-stride ensures adjacent threads always access adjacent float4 groups
 * → perfect coalescing on every load.
 *
 * Expected speedup sources (superlinear coupling):
 *   - 8× fewer iterations → 8× fewer R2 dependency episodes
 *   - 128-bit loads → 8× fewer MSHR entries (vs baseline)
 *   - 2 float4 loads per iteration → deep MLP within warp
 *
 * Usage: ./vectorAdd_vec4_unroll2 <numElements>
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

__global__ void vectorAdd_vec4_unroll2(const float4 *A, const float4 *B,
                                        float4 *C, long long numElements4) {
  long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
  long long gridStride = (long long)blockDim.x * gridDim.x;

  // Grid-stride loop with float4: adjacent threads access adjacent float4
  // groups → perfect 128-bit coalesced loads. 2× unroll for deeper MLP.
  #pragma unroll 2
  for (long long i = tid; i < numElements4; i += gridStride) {
    float4 a = A[i];
    float4 b = B[i];
    float4 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;
    C[i] = c;
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
  if (numElements % 4 != 0) {
    fprintf(stderr, "numElements must be divisible by 4 for float4 vectorization\n");
    return 1;
  }
  long long numElements4 = numElements / 4;
  size_t size = numElements * sizeof(float);
  double totalBytes = 3.0 * size;
  double totalFLOPs = (double)numElements;

  printf("=== vectorAdd_vec4_unroll2 (LDG.128 + grid-stride 2x unroll) ===\n");
  printf("Elements:  %lld (float4 groups: %lld, ~8 elements/thread)\n",
         numElements, numElements4);
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

  // Launch 1/2 of vec4 grid — each thread processes ~2 float4 groups = ~8 elements
  int threadsPerBlock = 256;
  long long totalThreads = (numElements4 + 1) / 2;
  int blocksPerGrid = (int)((totalThreads + threadsPerBlock - 1) / threadsPerBlock);
  printf("Launch: %d blocks x %d threads\n", blocksPerGrid, threadsPerBlock);

  // Warmup
  vectorAdd_vec4_unroll2<<<blocksPerGrid, threadsPerBlock>>>(
      (float4 *)d_A, (float4 *)d_B, (float4 *)d_C, numElements4);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timed run
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  vectorAdd_vec4_unroll2<<<blocksPerGrid, threadsPerBlock>>>(
      (float4 *)d_A, (float4 *)d_B, (float4 *)d_C, numElements4);
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
