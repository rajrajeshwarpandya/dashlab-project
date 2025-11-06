#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

#define ms 1024
#define bsize 32

__global__ void mat_mul_naive(const float *a, const float *b, float *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ms && col < ms) {
        float sum = 0.0f;
        for (int k = 0; k < ms; ++k)
            sum += a[row * ms + k] * b[k * ms + col];
        c[row * ms + col] = sum;
    }
}

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

    dim3 block(bsize, bsize);
    dim3 grid((ms + bsize - 1) / bsize, (ms + bsize - 1) / bsize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iterations = 100;
    float total_ms = 0.0f;

    cudaMemset(d_c, 0, bytes);
    mat_mul_naive<<<grid, block>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    total_ms = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        cudaMemset(d_c, 0, bytes);
        cudaEventRecord(start);
        mat_mul_naive<<<grid, block>>>(d_a, d_b, d_c);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_kernel = 0.0f;
        cudaEventElapsedTime(&ms_kernel, start, stop);
        total_ms += ms_kernel;
    }
    double avg_ms = (double)total_ms / iterations;
    double gflops = (2.0 * ms * ms * ms) / (avg_ms / 1e3) / 1e9;
    printf("\n1. Naive Kernel(100):\n", iterations);
    printf("   Time: %.3f ms\n", avg_ms);
    printf("   Performance: %.3f GFLOPS\n", gflops);

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