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

__global__ void mat_mul_reg_buffered(const float *a, const float *b, float *c) {
    
    __shared__ float as[2][bsize][bsize];
    __shared__ float bs[2][bsize][bsize];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int tid = ty * block_x + tx;
    int ib = blockIdx.y * bsize;
    int jb = blockIdx.x * bsize;

    float csub[reg_dim][reg_dim] = {{0.0f}};
    int num_tiles = ms / bsize;
    
    int ga_col = 0 * bsize;
    int gb_row = 0 * bsize;
    for (int k = 0; k < 16; ++k) {
        int idx = tid + k * 64;
        int sr = idx / 32;
        int sc = idx % 32;
        as[0][sr][sc] = a[(ib + sr) * ms + (ga_col + sc)];
        bs[0][sr][sc] = b[(gb_row + sr) * ms + (jb + sc)];
    }
    __syncthreads();

    for (int t = 1; t < num_tiles; t++) {
        
        int comp_buf = (t - 1) % 2;
        int load_buf = t % 2;
        
        ga_col = t * bsize;
        gb_row = t * bsize;
        for (int k = 0; k < 16; ++k) {
            int idx = tid + k * 64;
            int sr = idx / 32;
            int sc = idx % 32;
            as[load_buf][sr][sc] = a[(ib + sr) * ms + (ga_col + sc)];
            bs[load_buf][sr][sc] = b[(gb_row + sr) * ms + (jb + sc)];
        }

        for (int k = 0; k < bsize; k++) {
            float areg[reg_dim];
            float breg[reg_dim];
            for (int m = 0; m < reg_dim; m++) areg[m] = as[comp_buf][ty * reg_dim + m][k];
            for (int n = 0; n < reg_dim; n++) breg[n] = bs[comp_buf][k][tx * reg_dim + n];
            for (int m = 0; m < reg_dim; m++) {
                for (int n = 0; n < reg_dim; n++) {
                    csub[m][n] += areg[m] * breg[n];
                }
            }
        }
        
        __syncthreads();
    }

    int last_buf = (num_tiles - 1) % 2;
    for (int k = 0; k < bsize; k++) {
        float areg[reg_dim];
        float breg[reg_dim];
        for (int m = 0; m < reg_dim; m++) areg[m] = as[last_buf][ty * reg_dim + m][k];
        for (int n = 0; n < reg_dim; n++) breg[n] = bs[last_buf][k][tx * reg_dim + n];
        for (int m = 0; m < reg_dim; m++) {
            for (int n = 0; n < reg_dim; n++) {
                csub[m][n] += areg[m] * breg[n];
            }
        }
    }

    __shared__ float cs[bsize][bsize];
    for(int m = 0; m < reg_dim; m++) {
        for(int n = 0; n < reg_dim; n++) {
            cs[ty * reg_dim + m][tx * reg_dim + n] = csub[m][n];
        }
    }
    __syncthreads();
    
    for (int k = 0; k < 16; ++k) {
        int idx = tid + k * 64;
        int sr = idx / 32;
        int sc = idx % 32;
        c[(ib + sr) * ms + (jb + sc)] = cs[sr][sc];
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
    dim3 grid_reg(ms / bsize, ms / bsize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iters = 100;
    float total_ms = 0.0f;
    double gflops_base = (2.0 * ms * ms * ms) / 1e9;

    mat_mul_reg_buffered<<<grid_reg, block_reg>>>(d_a, d_b, d_c);
    cudaMemset(d_c, 0, bytes);
    total_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        mat_mul_reg_buffered<<<grid_reg, block_reg>>>(d_a, d_b, d_c);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms_kernel = 0.0f;
        cudaEventElapsedTime(&ms_kernel, start, stop);
        total_ms += ms_kernel;
    }
    printf("   Time: %.3f ms\n", (double)total_ms / iters);
    printf   ("   Performance: %.3f GFLOPS\n", gflops_base / ((double)total_ms / iters / 1e3));

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