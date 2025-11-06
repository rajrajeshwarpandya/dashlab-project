#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// --- Common Definitions ---
#define ms 1024         // Matrix Size (ms x ms)
#define bsize 32        // Block Size (bsize x bsize)

// --- Definitions for Register Tiling Kernels ---
#define reg_dim 4       // Register tile dimensions (reg_dim x reg_dim)
#define block_x 8       // Threads in x-dim for register kernels
#define block_y 8       // Threads in y-dim for register kernels (8x8 = 64 threads)

// --- Benchmarking Runs ---
#define warmup_runs 10
#define benchmark_runs 100

// ===================================================================
// KERNEL 1: Naive (from naive.cu)
// ===================================================================
__global__ void kernel_naive(const float *a, const float *b, float *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ms && col < ms) {
        float sum = 0.0f;
        for (int k = 0; k < ms; ++k)
            sum += a[row * ms + k] * b[k * ms + col];
        c[row * ms + col] = sum;
    }
}

// ===================================================================
// KERNEL 2: Tiled (from tiled.cu)
// ===================================================================
__global__ void kernel_tiled(const float *a, const float *b, float *c) {
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int i = by * blockDim.y + ty;
    int j = bx * blockDim.x + tx;
    
    __shared__ float as[bsize][bsize];
    __shared__ float bs[bsize][bsize];
    
    float sum = 0.0f;
    int num_tiles = (ms + bsize - 1) / bsize;
    
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * bsize + tx;
        int b_row = t * bsize + ty;

        if (i < ms && a_col < ms)
            as[ty][tx] = a[i * ms + a_col];
        else
            as[ty][tx] = 0.0f;

        if (b_row < ms && j < ms)
            bs[ty][tx] = b[b_row * ms + j];
        else
            bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < bsize; k++)
            sum += as[ty][k] * bs[k][tx];
        
        __syncthreads();
    }

    if (i < ms && j < ms)
        c[i * ms + j] = sum;
}

// ===================================================================
// KERNEL 3: Tiled + Double Buffering (from double_buffering.cu)
// ===================================================================
__global__ void kernel_tiled_double_buffered(const float *a, const float *b, float *c, int m)
{
    __shared__ float sh_a[2][bsize][bsize];
    __shared__ float sh_b[2][bsize][bsize];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int c_row = by * bsize + ty;
    int c_col = bx * bsize + tx;

    float acc = 0.0f;
    int num_tiles = m / bsize;

    if (c_row >= m || c_col >= m) return;

    sh_a[0][ty][tx] = a[c_row * m + tx];
    sh_b[0][ty][tx] = b[ty * m + c_col];
    __syncthreads();

    for (int k = 0; k < num_tiles - 1; ++k)
    {
        int current_buf = k % 2;
        int next_buf = (k + 1) % 2;

        sh_a[next_buf][ty][tx] = a[c_row * m + (k + 1) * bsize + tx];
        sh_b[next_buf][ty][tx] = b[((k + 1) * bsize + ty) * m + c_col];

        for (int i = 0; i < bsize; ++i)
        {
            acc += sh_a[current_buf][ty][i] * sh_b[current_buf][i][tx];
        }
        __syncthreads(); 
    }

    int last_buf = (num_tiles - 1) % 2;
    for (int i = 0; i < bsize; ++i)
    {
        acc += sh_a[last_buf][ty][i] * sh_b[last_buf][i][tx];
    }

    c[c_row * m + c_col] = acc;
}

// ===================================================================
// KERNEL 4: Register Tiled (Uncoalesced) (from registered.cu)
// ===================================================================
__global__ void kernel_reg_tiled(const float *a, const float *b, float *c) {
    
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

// ===================================================================
// KERNEL 5: Register Tiled + Coalesced Load (from co-registered.cu)
// ===================================================================
__global__ void kernel_reg_coalesced(const float *a, const float *b, float *c) {
    
    __shared__ float as[bsize][bsize];
    __shared__ float bs[bsize][bsize];
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int tid = ty * block_x + tx; 
    int ib = blockIdx.y * bsize; 
    int jb = blockIdx.x * bsize; 

    float csub[reg_dim][reg_dim] = {{0.0f}};
    int num_tiles = ms / bsize;
    
    for (int t = 0; t < num_tiles; t++) {
        
        int ga_col = t * bsize;
        int gb_row = t * bsize;
        
        for (int k = 0; k < 16; ++k) {
            int idx = tid + k * 64; 
            int sr = idx / 32;
            int sc = idx % 32;
            if((ib + sr) < ms && (ga_col + sc) < ms)
                as[sr][sc] = a[(ib + sr) * ms + (ga_col + sc)];
            else
                as[sr][sc] = 0.0f;
            
            if((gb_row + sr) < ms && (jb + sc) < ms)
                bs[sr][sc] = b[(gb_row + sr) * ms + (jb + sc)];
            else
                bs[sr][sc] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < bsize; k++) {
            float areg[reg_dim];
            float breg[reg_dim];
            for (int m = 0; m < reg_dim; m++) areg[m] = as[ty * reg_dim + m][k];
            for (int n = 0; n < reg_dim; n++) breg[n] = bs[k][tx * reg_dim + n];
            for (int m = 0; m < reg_dim; m++) {
                for (int n = 0; n < reg_dim; n++) {
                    csub[m][n] += areg[m] * breg[n];
                }
            }
        }
        __syncthreads();
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
        if((ib + sr) < ms && (jb + sc) < ms)
            c[(ib + sr) * ms + (jb + sc)] = cs[sr][sc];
    }
}

// ===================================================================
// KERNEL 6: Register Tiled + Coalesced + Double Buffered
// (from not_chosen.cu)
// ===================================================================
__global__ void kernel_reg_double_buffered(const float *a, const float *b, float *c) {
    
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
        if((ib + sr) < ms && (ga_col + sc) < ms)
            as[0][sr][sc] = a[(ib + sr) * ms + (ga_col + sc)];
        else
            as[0][sr][sc] = 0.0f;
        
        if((gb_row + sr) < ms && (jb + sc) < ms)
            bs[0][sr][sc] = b[(gb_row + sr) * ms + (jb + sc)];
        else
            bs[0][sr][sc] = 0.0f;
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
            if((ib + sr) < ms && (ga_col + sc) < ms)
                as[load_buf][sr][sc] = a[(ib + sr) * ms + (ga_col + sc)];
            else
                as[load_buf][sr][sc] = 0.0f;

            if((gb_row + sr) < ms && (jb + sc) < ms)
                bs[load_buf][sr][sc] = b[(gb_row + sr) * ms + (jb + sc)];
            else
                bs[load_buf][sr][sc] = 0.0f;
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
        if((ib + sr) < ms && (jb + sc) < ms)
            c[(ib + sr) * ms + (jb + sc)] = cs[sr][sc];
    }
}


// ===================================================================
// HELPER: Matrix Initialization
// ===================================================================
void init_matrix(float *m) {
    for (int i = 0; i < ms * ms; ++i)
        m[i] = (float)rand() / RAND_MAX;
}

// ===================================================================
// MAIN: Benchmarking
// ===================================================================
int main() {
    srand((unsigned int)time(NULL));
    size_t bytes = (size_t)ms * ms * sizeof(float);

    // --- 1. Allocate Host Memory ---
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes); 

    init_matrix(h_a);
    init_matrix(h_b);

    // --- 2. Allocate Device Memory ---
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // --- 3. Setup Benchmarking Tools ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    double gflops_base = (2.0 * (double)ms * (double)ms * (double)ms) / 1e9;
    double cublas_gflops = 0.0;
    
    float total_ms = 0.0f;
    float ms_kernel = 0.0f;
    double avg_ms = 0.0;
    double gflops = 0.0;
    double percent_of_cublas = 0.0;

    // --- Arrays to store results for the table ---
    const int num_kernels = 7;
    const char* kernel_names[num_kernels] = {
        "1. cuBLAS (Baseline)",
        "2. Naive",
        "3. Tiled",
        "4. Tiled + Double Buffered",
        "5. Register Tiled (Uncoalesced)",
        "6. Register Tiled (Coalesced)",
        "7. Reg. Tiled (Coal + Dbl. Buf)"
    };
    double times[num_kernels];
    double gflops_results[num_kernels];
    double percentages[num_kernels];


    printf("Starting benchmarks for %dx%d matrix multiplication...\n", ms, ms);
    printf("Averaging over %d runs after %d warmup runs.\n\n", benchmark_runs, warmup_runs);

    // --- 4. Define Grid/Block Dims ---
    dim3 block_std(bsize, bsize); 
    dim3 grid_std((ms + bsize - 1) / bsize, (ms + bsize - 1) / bsize);
    
    dim3 block_reg(block_x, block_y); 
    dim3 grid_reg_tiled(ms / (block_x * reg_dim), ms / (block_y * reg_dim)); 
    dim3 grid_reg_coalesced(ms / bsize, ms / bsize); 
    dim3 grid_tiled_db(ms / bsize, ms / bsize); 

    // ===================================
    // BENCHMARK 1: cuBLAS (Baseline)
    // ===================================
    printf("Benchmarking %s...", kernel_names[0]);
    total_ms = 0.0f;
    for (int i = 0; i < warmup_runs; ++i)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ms, ms, ms, &alpha, d_a, ms, d_b, ms, &beta, d_c, ms);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < benchmark_runs; ++i) {
        cudaMemset(d_c, 0, bytes);
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ms, ms, ms, &alpha, d_a, ms, d_b, ms, &beta, d_c, ms);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_kernel, start, stop);
        total_ms += ms_kernel;
    }
    avg_ms = (double)total_ms / benchmark_runs;
    cublas_gflops = gflops_base / (avg_ms / 1e3);
    
    times[0] = avg_ms;
    gflops_results[0] = cublas_gflops;
    percentages[0] = 100.0;
    printf(" Done.\n");

    // ===================================
    // BENCHMARK 2: Naive
    // ===================================
    printf("Benchmarking %s...", kernel_names[1]);
    total_ms = 0.0f;
    for (int i = 0; i < warmup_runs; ++i)
        kernel_naive<<<grid_std, block_std>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < benchmark_runs; ++i) {
        cudaMemset(d_c, 0, bytes);
        cudaEventRecord(start);
        kernel_naive<<<grid_std, block_std>>>(d_a, d_b, d_c);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_kernel, start, stop);
        total_ms += ms_kernel;
    }
    avg_ms = (double)total_ms / benchmark_runs;
    gflops = gflops_base / (avg_ms / 1e3);
    percent_of_cublas = (gflops / cublas_gflops) * 100.0;
    
    times[1] = avg_ms;
    gflops_results[1] = gflops;
    percentages[1] = percent_of_cublas;
    printf(" Done.\n");

    // ===================================
    // BENCHMARK 3: Tiled
    // ===================================
    printf("Benchmarking %s...", kernel_names[2]);
    total_ms = 0.0f;
    for (int i = 0; i < warmup_runs; ++i)
        kernel_tiled<<<grid_std, block_std>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < benchmark_runs; ++i) {
        cudaMemset(d_c, 0, bytes);
        cudaEventRecord(start);
        kernel_tiled<<<grid_std, block_std>>>(d_a, d_b, d_c);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_kernel, start, stop);
        total_ms += ms_kernel;
    }
    avg_ms = (double)total_ms / benchmark_runs;
    gflops = gflops_base / (avg_ms / 1e3);
    percent_of_cublas = (gflops / cublas_gflops) * 100.0;
    
    times[2] = avg_ms;
    gflops_results[2] = gflops;
    percentages[2] = percent_of_cublas;
    printf(" Done.\n");

    // ===================================
    // BENCHMARK 4: Tiled + Double Buffered
    // ===================================
    printf("Benchmarking %s...", kernel_names[3]);
    total_ms = 0.0f;
    for (int i = 0; i < warmup_runs; ++i)
        kernel_tiled_double_buffered<<<grid_tiled_db, block_std>>>(d_a, d_b, d_c, ms);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < benchmark_runs; ++i) {
        cudaMemset(d_c, 0, bytes);
        cudaEventRecord(start);
        kernel_tiled_double_buffered<<<grid_tiled_db, block_std>>>(d_a, d_b, d_c, ms);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_kernel, start, stop);
        total_ms += ms_kernel;
    }
    avg_ms = (double)total_ms / benchmark_runs;
    gflops = gflops_base / (avg_ms / 1e3);
    percent_of_cublas = (gflops / cublas_gflops) * 100.0;
    
    times[3] = avg_ms;
    gflops_results[3] = gflops;
    percentages[3] = percent_of_cublas;
    printf(" Done.\n");

    // ===================================
    // BENCHMARK 5: Register Tiled (Uncoalesced)
    // ===================================
    printf("Benchmarking %s...", kernel_names[4]);
    total_ms = 0.0f;
    for (int i = 0; i < warmup_runs; ++i)
        kernel_reg_tiled<<<grid_reg_tiled, block_reg>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < benchmark_runs; ++i) {
        cudaMemset(d_c, 0, bytes);
        cudaEventRecord(start);
        kernel_reg_tiled<<<grid_reg_tiled, block_reg>>>(d_a, d_b, d_c);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_kernel, start, stop);
        total_ms += ms_kernel;
    }
    avg_ms = (double)total_ms / benchmark_runs;
    gflops = gflops_base / (avg_ms / 1e3);
    percent_of_cublas = (gflops / cublas_gflops) * 100.0;
    
    times[4] = avg_ms;
    gflops_results[4] = gflops;
    percentages[4] = percent_of_cublas;
    printf(" Done.\n");

    // ===================================
    // BENCHMARK 6: Register Tiled (Coalesced)
    // ===================================
    printf("Benchmarking %s...", kernel_names[5]);
    total_ms = 0.0f;
    for (int i = 0; i < warmup_runs; ++i)
        kernel_reg_coalesced<<<grid_reg_coalesced, block_reg>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < benchmark_runs; ++i) {
        cudaMemset(d_c, 0, bytes);
        cudaEventRecord(start);
        kernel_reg_coalesced<<<grid_reg_coalesced, block_reg>>>(d_a, d_b, d_c);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_kernel, start, stop);
        total_ms += ms_kernel;
    }
    avg_ms = (double)total_ms / benchmark_runs;
    gflops = gflops_base / (avg_ms / 1e3);
    percent_of_cublas = (gflops / cublas_gflops) * 100.0;
    
    times[5] = avg_ms;
    gflops_results[5] = gflops;
    percentages[5] = percent_of_cublas;
    printf(" Done.\n");

    // ===================================
    // BENCHMARK 7: Register Tiled (Coalesced + Dbl. Buffered)
    // ===================================
    printf("Benchmarking %s...", kernel_names[6]);
    total_ms = 0.0f;
    for (int i = 0; i < warmup_runs; ++i)
        kernel_reg_double_buffered<<<grid_reg_coalesced, block_reg>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    
    for (int i = 0; i < benchmark_runs; ++i) {
        cudaMemset(d_c, 0, bytes);
        cudaEventRecord(start);
        kernel_reg_double_buffered<<<grid_reg_coalesced, block_reg>>>(d_a, d_b, d_c);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms_kernel, start, stop);
        total_ms += ms_kernel;
    }
    avg_ms = (double)total_ms / benchmark_runs;
    gflops = gflops_base / (avg_ms / 1e3);
    percent_of_cublas = (gflops / cublas_gflops) * 100.0;
    
    times[6] = avg_ms;
    gflops_results[6] = gflops;
    percentages[6] = percent_of_cublas;
    printf(" Done.\n\n");


    // ===================================
    // FINAL: Print Comparison Matrix
    // ===================================
    printf("+------------------------------------+-------------+-------------+---------------+\n");
    printf("| Kernel                             | Avg. Time   | Performance | % of cuBLAS   |\n");
    printf("|                                    | (ms)        | (GFLOPS)    | Performance   |\n");
    printf("+------------------------------------+-------------+-------------+---------------+\n");
    
    for (int i = 0; i < num_kernels; ++i) {
        printf("| %-34s | %11.3f | %11.3f | %13.2f%% |\n", 
               kernel_names[i], 
               times[i], 
               gflops_results[i], 
               percentages[i]);
        if (i == 0) {
            printf("+------------------------------------+-------------+-------------+---------------+\n");
        }
    }
    printf("+------------------------------------+-------------+-------------+---------------+\n");


    // --- 5. Cleanup ---
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