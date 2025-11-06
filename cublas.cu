#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <time.h>

#define ms 1024

void init_matrix(float *m) {
    for (int i = 0; i < ms * ms; ++i)
        m[i] = (float)rand() / RAND_MAX;
}

int main() {
    srand((unsigned int)time(NULL));
    size_t bytes = ms * ms * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    init_matrix(h_a);
    init_matrix(h_b);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    int iters = 100;
    float total_ms = 0.0f;
    double gflops_base = (2.0 * ms * ms * ms) / 1e9;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ms, ms, ms, &alpha, d_b, ms, d_a, ms, &beta, d_c, ms);
    
    cudaMemset(d_c, 0, bytes);
    total_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ms, ms, ms, &alpha, d_b, ms, d_a, ms, &beta, d_c, ms);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_kernel = 0.0f;
        cudaEventElapsedTime(&ms_kernel, start, stop);
        total_ms += ms_kernel;
    }
    
    printf("\n6. cuBLAS Kernel (Avg over %d runs):\n", iters);
    printf("   Time: %.3f ms\n", (double)total_ms / iters);
    printf("   Performance: %.3f GFLOPS\n", gflops_base / ((double)total_ms / iters / 1e3));

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}