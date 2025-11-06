#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

#define ms 1024
#define bsize 32

#define reg_dim 4
#define block_x 8
#define block_y 8

__global__ void mat_mul_reg_tiled(const float *a, const float *b, float *c) {
    
    __shared__ float as[bsize][bsize];
    __shared__ float bs[bsize][bsize];
    
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int i = (by * block_y + ty) * reg_dim;
    int j = (bx * block_x + tx) * reg_dim;

    float c_sub[reg_dim][reg_dim] = {{0.0f}};

    int num_tiles = (ms + bsize - 1) / bsize;
    
    for (int t = 0; t < num_tiles; t++) {
        
        for(int row = 0; row < reg_dim; row++) {
            for(int col = 0; col < reg_dim; col++) {
                
                int a_row = i + row;
                int a_col = t * bsize + tx * reg_dim + col;
                int as_row = ty * reg_dim + row;
                int as_col = tx * reg_dim + col;
                
                if (a_row < ms && a_col < ms)
                    as[as_row][as_col] = a[a_row * ms + a_col];
                else
                    as[as_row][as_col] = 0.0f;
                
                int b_row = t * bsize + ty * reg_dim + row;
                int b_col = j + col;
                int bs_row = ty * reg_dim + row;
                int bs_col = tx * reg_dim + col;

                if (b_row < ms && b_col < ms)
                    bs[bs_row][bs_col] = b[b_row * ms + b_col];
                else
                    bs[bs_row][bs_col] = 0.0f;
            }
        }
        __syncthreads();

        for (int k = 0; k < bsize; k++) {
            
            float a_reg[reg_dim];
            float b_reg[reg_dim];
            
            for (int m = 0; m < reg_dim; m++)
                a_reg[m] = as[ty * reg_dim + m][k];
                
            for (int n = 0; n < reg_dim; n++)
                b_reg[n] = bs[k][tx * reg_dim + n];
                
            for (int m = 0; m < reg_dim; m++) {
                for (int n = 0; n < reg_dim; n++) {
                    c_sub[m][n] += a_reg[m] * b_reg[n];
                }
            }
        }
        __syncthreads();
    }

    for(int m = 0; m < reg_dim; m++) {
        for(int n = 0; n < reg_dim; n++) {
            if ( (i + m) < ms && (j + n) < ms)
                c[(i + m) * ms + (j + n)] = c_sub[m][n];
        }
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

    dim3 block_reg(block_x, block_y);
    dim3 grid_reg(ms / (block_x * reg_dim), ms / (block_y * reg_dim));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iterations = 100;
    float total_ms = 0.0f;

    cudaMemset(d_c, 0, bytes);
    mat_mul_reg_tiled<<<grid_reg, block_reg>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    total_ms = 0.0f;
    for (int i = 0; i < iterations; ++i) {
        cudaMemset(d_c, 0, bytes);
        cudaEventRecord(start);
        mat_mul_reg_tiled<<<grid_reg, block_reg>>>(d_a, d_b, d_c);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_kernel = 0.0f;
        cudaEventElapsedTime(&ms_kernel, start, stop);
        total_ms += ms_kernel;
    }
    double avg_ms = (double)total_ms / iterations;
    double gflops = (2.0 * ms * ms * ms) / (avg_ms / 1e3) / 1e9;
    printf("\n3. Register Tiled Kernel (100):\n", 
           block_x, block_y, reg_dim, reg_dim, iterations);
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