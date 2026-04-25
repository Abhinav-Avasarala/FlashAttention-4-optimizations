#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// ============================================================================
// WHY THIS BENCHMARK IS DIFFERENT
//
// The standalone exp() kernels are memory-bandwidth bound: the GPU spends all
// its time loading/storing floats, not computing. Replacing __expf with a
// polynomial saves compute that was already hidden behind memory latency.
//
// Here we fix that by doing ITERS exp() calls per element, all in registers,
// before writing once. This makes exp() the actual bottleneck, which is the
// same situation as inside a fused attention kernel.
// ============================================================================

#define ITERS 512   // exp() calls per thread — enough to be compute-bound

// Hardware __expf: one multiply + one hardware ex2 instruction (~18 cycles)
__global__ void exp_loop_hardware(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = input[idx];
    float acc = 0.0f;

    #pragma unroll 8
    for (int i = 0; i < ITERS; i++) {
        acc += __expf(x);
        x = acc * 9.54e-7f;  // keep x in [-1, 1], prevent dead-code elimination
    }
    output[idx] = acc;
}

// Polynomial degree 4 using Horner's method — 4 FMAs (~16 cycles)
// No range reduction needed: inputs stay in [-1, 1] due to the feedback loop,
// where degree-4 Taylor error is <0.5% (acceptable for attention).
__device__ inline float poly4_horner(float x) {
    // exp(x) ≈ 1 + x(1 + x(1/2 + x(1/6 + x/24)))
    // Horner form minimizes multiplications vs expanded form
    return 1.0f + x * (1.0f + x * (0.5f + x * (0.16666667f + x * 0.041666667f)));
}

__global__ void exp_loop_poly4(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = input[idx];
    float acc = 0.0f;

    #pragma unroll 8
    for (int i = 0; i < ITERS; i++) {
        acc += poly4_horner(x);
        x = acc * 9.54e-7f;
    }
    output[idx] = acc;
}

int main() {
    int n = 1024 * 1024;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    printf("Compute-Bound Exp Benchmark\n");
    printf("============================\n");
    printf("Elements: %d  |  Exp calls per thread: %d  |  Total exp calls: %lld\n\n",
           n, ITERS, (long long)n * ITERS);

    // Setup
    float *h_input = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++)
        h_input[i] = -0.5f + (float)rand() / RAND_MAX;  // [-0.5, 0.5]

    float *d_input, *d_output;
    cudaMalloc(&d_input,  n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int warmup = 3, iters = 20;

    // ---- Hardware baseline ----
    for (int i = 0; i < warmup; i++)
        exp_loop_hardware<<<blocks, threads>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        exp_loop_hardware<<<blocks, threads>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_hw;
    cudaEventElapsedTime(&time_hw, start, stop);
    time_hw /= iters;

    // ---- Polynomial degree 4 ----
    for (int i = 0; i < warmup; i++)
        exp_loop_poly4<<<blocks, threads>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        exp_loop_poly4<<<blocks, threads>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_poly;
    cudaEventElapsedTime(&time_poly, start, stop);
    time_poly /= iters;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Hardware __expf:      %.4f ms\n", time_hw);
    printf("Polynomial degree 4:  %.4f ms\n", time_poly);
    printf("Speedup:              %.2fx\n\n", time_hw / time_poly);

    if (time_hw / time_poly > 1.05f)
        printf("Result: polynomial IS faster in compute-bound regime\n");
    else if (time_hw / time_poly < 0.95f)
        printf("Result: hardware __expf is faster — GPU exp instruction wins\n");
    else
        printf("Result: roughly equal — both hit the same hardware throughput ceiling\n");

    printf("\nNote: speedup here mirrors what Flash Attention 4 sees because\n");
    printf("exp() runs on register-resident data, not from global memory.\n");

    free(h_input);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
