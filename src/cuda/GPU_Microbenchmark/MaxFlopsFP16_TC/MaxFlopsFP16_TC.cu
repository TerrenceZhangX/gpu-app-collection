#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "mma.h"
#include <iostream>
#include "time.h"


#define V100_SM_NUM 80
#define V100_CLK_MHZ 1132.0

static __shared__ float tensor_out;
static __global__ void tensor_f16f16f32_hammer_kernel()
{
	__shared__ unsigned int smem[16];
    __half2 *A = (__half2 *)&smem[0];
    A[0] = __float2half2_rn(0.01);
    A[1] = __float2half2_rn(0.01);
    A[2] = __float2half2_rn(0.01);
    A[3] = __float2half2_rn(0.01);
    A[4] = __float2half2_rn(0.01);
    A[5] = __float2half2_rn(0.01);
    A[6] = __float2half2_rn(0.01);
    A[7] = __float2half2_rn(0.01);

    float *C0 = (float *)&smem[8];
    C0[0] = 0.01;
    C0[1] = 0.01;
    C0[2] = 0.01;
    C0[3] = 0.01;
    C0[4] = 0.01;
    C0[5] = 0.01;
    C0[6] = 0.01;
    C0[7] = 0.01;
    
    __syncthreads();

    for (int it = 0; it < 102400; ++it) {
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            asm (
                "wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 "
                "   { %0, %1, %2, %3, %4, %5, %6, %7 }, "
                "   { %8, %9,%10,%11,%12,%13,%14,%15 }, "
                "   { %8, %9,%10,%11,%12,%13,%14,%15 }, "
                "   { %0, %1, %2, %3, %4, %5, %6, %7 }; "
                : "+f"(C0[0]), "+f"(C0[1]), "+f"(C0[2]), "+f"(C0[3]),
                  "+f"(C0[4]), "+f"(C0[5]), "+f"(C0[6]), "+f"(C0[7])
                : "r"(smem[0]), "r"(smem[1]), "r"(smem[2]), "r"(smem[3]),
                  "r"(smem[4]), "r"(smem[5]), "r"(smem[6]), "r"(smem[7]),
                  "r"(smem[8]), "r"(smem[9]), "r"(smem[10]), "r"(smem[11]),
                  "r"(smem[12]), "r"(smem[13]), "r"(smem[14]), "r"(smem[15])
            );
        }
    }

    // Avoid compiler optimization. Force write back to HBM.
    // Write only takes a small portion in whole execution.
    if (C0[0] > 0.2) {
        tensor_out = C0[0] + C0[1] + C0[2] + C0[3] +
                     C0[4] + C0[5] + C0[6] + C0[7];
    }
}

int main() {
	dim3 grid = dim3(1, 1, 1);
    dim3 block = dim3(128, 1, 1);
    
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	cudaEventRecord(start);
    tensor_f16f16f32_hammer_kernel<<<grid,block>>>();
	cudaEventRecord(stop);
	cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

	double flops = 102400*32/V100_CLK_MHZ*(2*16*16*16*2);

	printf("Elapsed Time = %f ms\n",milliseconds);

    // In Volta GV100, each Tensor Core performs 64 floating point FMA
    // operations per clock, and eight Tensor Cores in an SM perform a total of 512 FMA operations (or
    // 1024 individual floating point operations) per clock
    // Spec val: FLOPS = 1024 flop/clk/SM
	printf("FLOPS = %f flop/clk/SM\n", flops/milliseconds/1e3);
	printf("Achieved FLOPS = %f TFLOPS\n", (flops/1e12)/(milliseconds/1e3)*V100_SM_NUM*V100_CLK_MHZ);
	
    return 0;
}