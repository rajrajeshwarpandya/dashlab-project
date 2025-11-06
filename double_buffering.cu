#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define tile_dim 32
#define n 1024
#define warmup_runs 10
#define benchmark_runs 100

__global__ void matMulDoubleBuffered(const float *a, const float *b, float *c, int m)
{
    __shared__ float sh_a[2][tile_dim][tile_dim];
    __shared__ float sh_b[2][tile_dim][tile_dim];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int c_row = by * tile_dim + ty;
    int c_col = bx * tile_dim + tx;

    float acc = 0.0f;
    int num_tiles = m / tile_dim;

    if (c_row >= m || c_col >= m) return;

    sh_a[0][ty][tx] = a[c_row * m + tx];
    sh_b[0][ty][tx] = b[ty * m + c_col];
    __syncthreads();

    for (int k = 0; k < num_tiles - 1; ++k)
    {
        int current_buf = k % 2;
        int next_buf = (k + 1) % 2;

        sh_a[next_buf][ty][tx] = a[c_row * m + (k + 1) * tile_dim + tx];
        sh_b[next_buf][ty][tx] = b[((k + 1) * tile_dim + ty) * m + c_col];

        for (int i = 0; i < tile_dim; ++i)
        {
            acc += sh_a[current_buf][ty][i] * sh_b[current_buf][i][tx];
        }
        __syncthreads();
    }

    int last_buf = (num_tiles - 1) % 2;
    for (int i = 0; i < tile_dim; ++i)
    {
        acc += sh_a[last_buf][ty][i] * sh_b[last_buf][i][tx];
    }

    c[c_row * m + c_col] = acc;
}

void init_matrix(float *m, int size)
{
    for (int i = 0; i < size; ++i)
    {
        m[i] = (float)rand() / (float)RAND_MAX;
    }
}

int main(void)
{
 
    size_t bytes = (size_t)n * n * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);

    srand(time(NULL));
    init_matrix(h_a, n * n);
    init_matrix(h_b, n * n);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(tile_dim, tile_dim);
    dim3 blocksPerGrid(n / tile_dim, n / tile_dim);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup_runs; ++i)
    {
        matMulDoubleBuffered<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    }
    cudaDeviceSynchronize();

    float total_time_ms = 0.0f;
    for (int i = 0; i < benchmark_runs; ++i)
    {
        cudaEventRecord(start);
        matMulDoubleBuffered<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        total_time_ms += elapsed_ms;
    }

    cudaDeviceSynchronize();

    double avg_time_ms = total_time_ms / benchmark_runs;
    double avg_time_s = avg_time_ms / 1000.0;

    double flops = 2.0 * (double)n * (double)n * (double)n;
    double gflops = (flops / avg_time_s) / 1e9;
    printf("Avg Time: %.4f ms\n", avg_time_ms);
    printf("GFLOPS: %.4f\n", gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);

    cudaDeviceReset();

    return 0;
}