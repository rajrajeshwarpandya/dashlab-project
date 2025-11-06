#  CUDA Matrix Multiplication Performance Report

## 1. Implementation Details

This section details the theory and implementation of each optimization step, starting from a naive kernel and progressively building to more complex, high-performance versions.

### 1.1 Kernel 1: Naive (`naive.cu`)

* **Description:** This is the most straightforward implementation. Each thread in the grid is responsible for computing exactly one element of the output matrix $c$.
* **Theory:** To compute $c[row][col]$, a thread must loop $ms$ (1024) times. In each iteration, it reads one element from matrix $a$ and one from matrix $b$, multiplies them, and adds to a sum. This results in $2 \times ms$ global memory reads for every *single* output element.
* **Bottleneck:** This kernel is entirely bound by **global memory latency and bandwidth**.
    1.  **No Data Reuse:** For a $1024 \times 1024$ matrix, $a[row][0]$ is read 1024 times (by all threads in that row), and $b[0][col]$ is read 1024 times (by all threads in that column). This is $O(n^3)$ global memory accesses.
    2.  **Uncoalesced Access:** Threads in a warp (32 threads) execute in lockstep. For `a[row * ms + k]`, threads with adjacent `threadIdx.y` values (in the same `col`) will access `a` rows apart, which is good (coalesced). However, for `b[k * ms + col]`, threads with adjacent `threadIdx.x` values (in the same `row`) access `b` columns apart. This is highly uncoalesced, as the 32 threads will access 32 different memory locations far apart, leading to 32 separate memory transactions.
* **Code Snippet:**
    ```c++
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
    ```

---

### 1.2 Kernel 2: Tiled (Shared Memory) (`tiled.cu`)

* **Description:** This kernel introduces **tiling**, the most fundamental optimization for matrix multiplication. The grid is divided into blocks, and each block is responsible for computing a $bsize \times bsize$ (32x32) tile of the output matrix $c$.
* **Theory:** It uses **shared memory**, which is a small, on-chip memory (like an L1 cache) that is extremely fast and shared by all threads in a block.
    1.  The kernel loops through the $a$ and $b$ matrices in "tile" steps.
    2.  In each step, all threads in the block work together to load one tile from $a$ and one from $b$ into two shared memory arrays, `as` and `bs`.
    3.  A `__syncthreads()` barrier ensures all data is loaded *before* any thread starts computing.
    4.  Threads then compute the dot product for their assigned $c$ element, but read *exclusively from fast shared memory*.
    5.  This reduces the number of global memory reads from $O(n^3)$ to $O(n^3 / \text{bsize})$, a massive improvement. Each element is read from global memory only $ms / bsize$ times.
* **Bottleneck:** While much faster, it's now limited by shared memory bandwidth and arithmetic latency. It's also still making many redundant reads from shared memory.
* **Code Snippet:**
    ```c++
    __shared__ float as[bsize][bsize];
    __shared__ float bs[bsize][bsize];
    
    // ... loop 't' for number of tiles ...
        // Load one tile into as and bs from global memory
        as[ty][tx] = a[...];
        bs[ty][tx] = b[...];
        __syncthreads();

        // Compute from shared memory
        for (int k = 0; k < bsize; k++)
            sum += as[ty][k] * bs[k][tx];
        __syncthreads();
    // ... end loop ...
    ```

---

### 1.3 Kernel 3: Tiled + Double Buffering (`double_buffering.cu`)

* **Description:** This kernel builds on the tiled approach to address a remaining bottleneck: the stall during memory loads. In Kernel 2, the entire block stops at `__syncthreads()` to wait for data to be loaded from global to shared memory.
* **Theory:** **Double buffering** is a technique to hide memory latency. It uses *two* sets of shared memory arrays (e.g., `sh_a[2][][]`).
    1.  The kernel loads the *first* tile (tile 0) into `sh_a[0]` and `sh_b[0]`.
    2.  It then enters the main loop. In iteration `k`, it:
        * **Loads** data for tile `k+1` into the *next* buffer (`sh_a[(k+1)%2]`).
        * **Computes** using data from the *current* buffer (`sh_a[k%2]`).
    3.  By issuing the load for the next tile *while* computing on the current one, the memory access latency is overlapped with computation, effectively "hiding" it.
* **Bottleneck:** This adds complexity. It requires more shared memory and more complex loop control. It effectively hides latency but doesn't increase the number of calculations (FLOPs) per cycle.
* **Code Snippet:**
    ```c++
    __shared__ float sh_a[2][tile_dim][tile_dim];
    __shared__ float sh_b[2][tile_dim][tile_dim];
    
    // ... preload first tile into buffer 0 ...
    __syncthreads();

    for (int k = 0; k < num_tiles - 1; ++k)
    {
        int current_buf = k % 2;
        int next_buf = (k + 1) % 2;

        // Load for k+1 into next_buf
        sh_a[next_buf][ty][tx] = a[...];
        sh_b[next_buf][ty][tx] = b[...];

        // Compute on k from current_buf
        for (int i = 0; i < tile_dim; ++i)
        {
            acc += sh_a[current_buf][ty][i] * sh_b[current_buf][i][tx];
        }
        __syncthreads(); // Wait for both load and compute to finish
    }
    // ... compute on last buffer ...
    ```

---

### 1.4 Kernel 4: Tiled + Register Blocking (`registered.cu`)

* **Description:** This kernel explores a different optimization path: **register blocking** (or "register tiling"). Instead of each thread computing one $c$ element, each thread now computes a small `reg_dim x reg_dim` (4x4) sub-matrix.
* **Theory:** The $4 \times 4 = 16$ accumulator values (`c_sub`) are stored in the thread's private **registers**, which are the fastest possible memory on the GPU.
    1.  In the inner compute loop (over `k`), each thread loads `reg_dim` (4) values from `as` into a register array (`a_reg`) and `reg_dim` (4) values from `bs` (`b_reg`).
    2.  It then performs a $4 \times 4$ "outer product" on these register-bound values, performing 16 multiply-adds.
    3.  This dramatically increases **arithmetic intensity**. For only $4+4=8$ reads from shared memory, it performs 16 FLOPs. This helps keep the GPU's math units (ALUs) saturated.
* **Challenge (Uncoalesced Loads):** This specific implementation (`registered.cu`) has a major flaw. The loop used to load data from global memory into shared memory is *uncoalesced*. Each thread loads 16 elements for its $4 \times 4$ tile, but it does so non-contiguously, re-introducing a global memory bottleneck.
* **Code Snippet:**
    ```c++
    float c_sub[reg_dim][reg_dim] = {{0.0f}};
    
    // ... loop over tiles 't' ...
        // !! UNCOALESCED LOAD !!
        for(int row = 0; row < reg_dim; row++) {
            for(int col = 0; col < reg_dim; col++) {
                as[...][...] = a[...];
                bs[...][...] = b[...];
            }
        }
        __syncthreads();

        for (int k = 0; k < bsize; k++) {
            float a_reg[reg_dim];
            float b_reg[reg_dim];
            // Load from shared to registers
            for (int m = 0; m < reg_dim; m++) a_reg[m] = as[ty * reg_dim + m][k];
            for (int n = 0; n < reg_dim; n++) b_reg[n] = bs[k][tx * reg_dim + n];
            
            // Compute in registers
            for (int m = 0; m < reg_dim; m++) {
                for (int n = 0; n < reg_dim; n++) {
                    c_sub[m][n] += a_reg[m] * b_reg[n];
                }
            }
        }
    // ... end loops ...
    ```

---

### 1.5 Kernel 5: Tiled + Register Blocking + Coalescing (`co-registered.cu`)

* **Description:** This kernel is the critical fix for Kernel 4. It keeps the highly efficient register blocking for *computation* but completely revamps the global-to-shared memory *load* to be fully **coalesced**.
* **Theory:** Instead of a 2D loop, the 64 threads in the block (8x8) are treated as a 1D group (`tid`). This group is responsible for loading the $32 \times 32 = 1024$ elements of a tile.
    1.  The load is split into 16 chunks (since $1024 / 64 = 16$).
    2.  In each chunk `k`, thread `tid` loads `as` and `bs` elements at an index calculated from `tid + k * 64`.
    3.  When a warp of 32 threads (e.g., `tid` 0-31) executes this, they access 32 consecutive memory addresses, resulting in a single, perfectly coalesced memory transaction.
* **Bottleneck:** This is a very fast kernel. The loading is efficient, and the computation has high arithmetic intensity. The remaining bottleneck is largely the instruction overhead and keeping the complex pipeline perfectly full.
* **Code Snippet:**
    ```c++
    // COALESCED LOAD
    for (int k = 0; k < 16; ++k) {
        int idx = tid + k * 64; // 64 threads in block
        int sr = idx / 32;       // Shared mem row
        int sc = idx % 32;       // Shared mem col
        as[sr][sc] = a[(ib + sr) * ms + (ga_col + sc)];
        bs[sr][sc] = b[(gb_row + sr) * ms + (jb + sc)];
    }
    __syncthreads();

    // Compute (identical to Kernel 4)
    for (int k = 0; k < bsize; k++) {
        // ... (load to a_reg, b_reg, compute on csub) ...
    }
    ```

---

### 1.6 Kernel 6: (Experiment) Reg + Coalesced + DB (`not_chosen.cu`)

* **Description:** This kernel is the final experiment, an attempt to combine the two most powerful techniques: the coalesced register blocking from Kernel 5 and the double buffering from Kernel 3.
* **Theory:** The goal was to achieve the "best of both worlds": use double buffering (`as[2][][]`) to hide the latency of the (already fast) coalesced load, while simultaneously using register tiling (`csub[][]`) to maximize arithmetic intensity.
* **Challenge (Diminishing Returns):** As the performance analysis shows, this kernel was actually *slower* than Kernel 5.
    * **Instruction Overhead:** The logic to manage the `load_buf` and `comp_buf` indices adds extra instructions to the inner loop, which is the most critical part of the kernel.
    * **Latency Already Hidden:** Register blocking (Kernel 5) is already so effective at hiding latency—not by overlapping (like double buffering), but by giving the GPU *so much math to do* (the $4 \times 4$ outer product) that the pipeline stays full, and the next memory load is ready by the time it's needed (this is called **Instruction-Level Parallelism**).
    * In this case, double buffering's overhead was greater than its benefit, as there was little latency left to hide.
* **Code Snippet:**
    ```c++
    __shared__ float as[2][bsize][bsize]; // Double buffer
    __shared__ float bs[2][bsize][bsize];
    float csub[reg_dim][reg_dim] = {{0.0f}}; // Register tiling

    // ... pre-load buffer 0 ...
    
    for (int t = 1; t < num_tiles; t++) {
        int comp_buf = (t - 1) % 2;
        int load_buf = t % 2;
        
        // Load into next buffer
        ... as[load_buf][sr][sc] = a[...]; ...

        // Compute from current buffer
        for (int k = 0; k < bsize; k++) {
            ... csub[m][n] += areg[m] * breg[n]; ...
        }
        __syncthreads();
    }
    // ... compute on last buffer ...
    ```

---

### 1.7 Reference: cuBLAS (`cublas.cu`)

* **Description:** This uses `cublasSgemm` from NVIDIA's cuBLAS library.
* **Theory:** This is not a single kernel, but a library function that calls a "black box" kernel. This kernel is hand-tuned by NVIDIA engineers for a specific GPU architecture. It likely uses:
    * **TF32 Tensor Cores** (on Ampere+ GPUs) for mixed-precision-like speedup on FP32 data.
    * Warp-level primitives and asynchronous prefetching.
    * A sophisticated auto-tuner that picks the *perfect* tile sizes, register blocking dimensions, and loop unrolling for the $1024 \times 1024$ problem size on your specific hardware.
* It serves as the "gold standard" or 100% baseline for theoretical performance.

---

### 1.8 Compilation Commands

To compile these kernels, a command similar to the following would be used:

```bash
# For all kernels except cublas
# Use an arch flag appropriate for your GPU (e.g., sm_86 for RTX 30-series)
nvcc -o tiled_kernel tiled.cu -O3 -arch=sm_86

# For the cublas kernel, you must link the library
nvcc -o cublas_kernel cublas.cu -O3 -arch=sm_86 -lcublas
```

* `nvcc`: The NVIDIA CUDA C/C++ compiler.
* `-o <name>`: Specifies the output executable file name.
* `<file.cu>`: The source file to compile.
* `-O3`: Enables high-level compiler optimizations.
* `-arch=sm_86`: Specifies the target GPU architecture (e.g., `sm_86` for an NVIDIA RTX 30-series card, `sm_75` for Turing, `sm_89` for Hopper). This is crucial for performance.
* `-lcublas`: Links the cuBLAS library (only needed for `cublas.cu`).

---

## 2. Performance Analysis

### 2.1 Performance Comparison

Here is the performance data for all implementations, based on a 100-run average for a $1024 \times 1024$ matrix multiplication.

*Note: Performance (GFLOPS) is calculated as $(2 \times N^3) / \text{Time (sec)}$, with $N=1024$. Speedup is relative to the `Naive` kernel.*

| # | Implementation | Time (ms) | GFLOPS | % of cuBLAS | Speedup vs Naive |
|---|---|---|---|---|---|
| 1 | cuBLAS (Baseline) | 1.331 | 1613.633 | 100.00% | 4.28x |
| 2 | Naive | 5.693 | 377.227 | 23.38% | 1.00x |
| 3 | Tiled (Shared Mem) | 4.442 | 483.468 | 29.96% | 1.28x |
| 4 | Tiled + Double Buffering | 4.327 | 496.260 | 30.75% | 1.32x |
| 5 | Tiled + Reg. Blocking (Uncoalesced) | 2.524 | 850.861 | 52.73% | 2.26x |
| 6 | **Tiled + Reg. Blocking + Coalesced** | **1.939** | **1107.645** | **68.64%** | **2.94x** |
| 7 | Reg + Coalesced + DB (Experiment) | 2.266 | 947.635 | 58.73% | 2.51x |


### 2.2 Performance Graph

(You would insert your bar chart or line plot here)

``

The chart would show:
* A base-level performance for "Naive".
* A small jump for "Tiled" and "Tiled + DB".
* A significant jump for "Tiled + Reg. Blocking (Uncoalesced)".
* Another large jump for "Tiled + Reg. Blocking (Coalesced)", which is the peak for all custom kernels.
* A "Reg + Coalesced + DB" bar that is noticeably *lower* than the previous one.
* A "cuBLAS" bar that is the highest of all, showing the reference performance.

---

### 2.3 Kernel-by-Kernel Analysis

1.  **Naive (1.00x) to Tiled (1.28x):**
    * **Bottleneck Addressed:** $O(N^3)$ global memory accesses.
    * **Result:** A 28% speedup. This shows that reducing global memory reads by using shared memory provides a solid, though not transformative, initial improvement.

2.  **Tiled (1.28x) to Tiled + Double Buffering (1.32x):**
    * **Bottleneck Addressed:** Global-to-shared memory load latency.
    * **Result:** A very small 3% improvement over the standard tiled kernel. This confirms the theory (overlapping memory and compute) but shows that on this hardware, the stall time was not the primary bottleneck.

3.  **Tiled (1.28x) to Reg. Blocking (Uncoalesced) (2.26x):**
    * **Bottleneck Addressed:** Shared memory bandwidth and arithmetic (math) pipeline utilization.
    * **Result:** A 77% improvement over the standard tiled kernel. This is a *major* gain, proving that increasing arithmetic intensity (FLOPs per memory op) by using registers is far more effective than just hiding latency.
    * **New Bottleneck:** This kernel introduced a *new* problem: the uncoalesced global load, which capped its potential.

4.  **Reg. Blocking (Uncoalesced) (2.26x) to Reg. Blocking + Coalesced (2.94x):**
    * **Bottleneck Addressed:** Uncoalesced global memory loads.
    * **Result:** A 30% speedup over its uncoalesced predecessor. This was the "fix" that unlocked the full potential of register blocking. It highlights that **memory coalescing is not optional**; it is a critical requirement for any high-performance kernel.

5.  **Analysis of the Experiment (Kernel 6 vs. Kernel 7):**
    * **Hypothesis:** Combining the fastest load-hiding technique (DB) with the fastest computation technique (Reg. Blocking) should be the fastest.
    * **Result:** Kernel 7 (2.51x) was ~15% *slower* than Kernel 6 (2.94x).
    * **Analysis (Diminishing Returns):** This is a perfect example of diminishing returns and optimization conflict. The register blocking in Kernel 6 is already *so effective* at keeping the compute units fed (by hiding latency with instruction-level parallelism) that there is no remaining memory latency to "hide" with double buffering. The extra instructions required to manage the double buffer (`comp_buf`, `load_buf`) just added overhead, slowing the kernel down.

### 2.4 Final Conclusion

This project successfully optimized a matrix multiplication kernel from 377.2 GFLOPS to 1107.6 GFLOPS, a **2.94x speedup** over the naive implementation.

The key takeaways are:
1.  **Register blocking** provided the most significant speedup, as it dramatically increases arithmetic intensity and GPU core utilization.
2.  **Memory coalescing** is absolutely critical. Fixing the uncoalesced load gave a 30% performance boost, proving that an uncoalesced load can completely undermine an otherwise-clever algorithm.
3.  **Tiling to shared memory** is a fundamental optimization, but its benefit can be overshadowed by other bottlenecks.
4.  **Optimizations are not always cumulative.** The double-buffered kernel was slower, showing that adding "optimizations" can add overhead and hurt performance if they don't address the *true* bottleneck.
5.  **cuBLAS is fast.** The final custom kernel achieved **68.6%** of cuBLAS performance, which is a very strong result. The remaining gap is due to architecture-specific tuning (like Tensor Core use and instruction scheduling) that goes beyond general optimization principles.
