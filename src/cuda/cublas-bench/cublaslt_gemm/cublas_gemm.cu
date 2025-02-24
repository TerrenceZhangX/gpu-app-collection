#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>
#include <getopt.h>
#include <cstdlib>

// Define argument structure
struct Args {
    int m = 1024;
    int n = 1024;
    int k = 1024;
    int batch = 1;
    int warmup = 5;
    int iter = 10;
};

// Parse command-line arguments
void process_args(int argc, char **argv, Args *args) {
    const char *const short_opts = "m:n:k:b:w:i:";
    const option long_opts[] = {
        {"batch", required_argument, nullptr, 'b'},
        {"warmup", required_argument, nullptr, 'w'},
        {"iter", required_argument, nullptr, 'i'},
        {nullptr, 0, nullptr, 0}
    };

    int opt = 0;
    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'm': args->m = std::stoi(optarg); break;
            case 'n': args->n = std::stoi(optarg); break;
            case 'k': args->k = std::stoi(optarg); break;
            case 'b': args->batch = std::stoi(optarg); break;
            case 'w': args->warmup = std::stoi(optarg); break;
            case 'i': args->iter = std::stoi(optarg); break;
            default: throw std::invalid_argument("Invalid argument!");
        }
    }
}

// Initialize matrix with random values
void initialize_matrix(half* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// GEMM computation using cublasGemmEx
void run_gemm(const Args &args) {
    int m = args.m, n = args.n, k = args.k;
    size_t matrix_size_a = m * k * sizeof(half);
    size_t matrix_size_b = k * n * sizeof(half);
    size_t matrix_size_c = m * n * sizeof(half);

    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, matrix_size_a);
    cudaMalloc(&d_B, matrix_size_b);
    cudaMalloc(&d_C, matrix_size_c);

    half *h_A = (half *)malloc(matrix_size_a);
    half *h_B = (half *)malloc(matrix_size_b);
    initialize_matrix(h_A, m * k);
    initialize_matrix(h_B, k * n);

    cudaMemcpy(d_A, h_A, matrix_size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrix_size_b, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    const half alpha = 1.0f;
    const half beta = 0.0f;

    // Warm-up iterations
    for (int i = 0; i < args.warmup; ++i) {
        cublasGemmEx(handle,
                     CUBLAS_OP_N, CUBLAS_OP_N, 
                     m, n, k, 
                     &alpha, d_A, CUDA_R_16F, m, 
                     d_B, CUDA_R_16F, k, 
                     &beta, d_C, CUDA_R_16F, m, 
                     CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();

    // Timing run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < args.iter; ++i) {
        cublasGemmEx(handle,
                     CUBLAS_OP_N, CUBLAS_OP_N, 
                     m, n, k, 
                     &alpha, 
                     d_A, CUDA_R_16F, m, 
                     d_B, CUDA_R_16F, k, 
                     &beta, 
                     d_C, CUDA_R_16F, m, 
                     CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float time_us = milliseconds / args.iter * 1e3;

    // Print the results
    printf("M\tN\tK\tB\tTime(us)\tAchieved TFLOPS\n");
    printf("%d\t%d\t%d\t%d\t%f\t%f\n", args.m, args.n, args.k, args.batch, time_us,
           2 * float(args.m) * float(args.n) * float(args.k) / 1e6 / time_us);

    // Free resources
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    cublasDestroy(handle);
}

int main(int argc, char **argv) {
    Args args;
    process_args(argc, argv, &args);
    run_gemm(args);
    return 0;
}
