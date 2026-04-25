#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <algorithm>

// ============================================================================
// HARDWARE EXPONENTIAL KERNEL (baseline)
// ============================================================================
__global__ void exp_hardware(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __expf(input[idx]);
    }
}

// ============================================================================
// POLYNOMIAL APPROXIMATION KERNELS
// ============================================================================

// Range reduction helper: exp(x) = 2^k * exp(r), r in [-ln2/2, ln2/2]
// This keeps r small so the Taylor series converges accurately.
__device__ inline float exp_range_reduce(float x, float poly_r) {
    // Clamp to avoid overflow/underflow in 2^k
    x = fmaxf(fminf(x, 88.0f), -88.0f);
    const float inv_ln2 = 1.44269504f;
    const float ln2    = 0.69314718f;
    int k = __float2int_rn(x * inv_ln2);   // round to nearest int
    float r = x - k * ln2;                 // residual, |r| <= ln2/2 ~ 0.347
    // poly_r is the caller's polynomial evaluated at r
    // Multiply by 2^k via IEEE 754 exponent field
    return poly_r * __int_as_float((k + 127) << 23);
}

// Degree 3: exp(r) ≈ 1 + r + r²/2 + r³/6  (accurate for small r)
__global__ void exp_poly3(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        const float inv_ln2 = 1.44269504f, ln2 = 0.69314718f;
        x = fmaxf(fminf(x, 88.0f), -88.0f);
        int k = __float2int_rn(x * inv_ln2);
        float r = x - k * ln2;
        float poly = 1.0f + r + r*r*0.5f + r*r*r*0.16666667f;
        output[idx] = poly * __int_as_float((k + 127) << 23);
    }
}

// Degree 4: exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24
__global__ void exp_poly4(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        const float inv_ln2 = 1.44269504f, ln2 = 0.69314718f;
        x = fmaxf(fminf(x, 88.0f), -88.0f);
        int k = __float2int_rn(x * inv_ln2);
        float r = x - k * ln2;
        float r2 = r * r;
        float poly = 1.0f + r + r2*0.5f + r2*r*0.16666667f + r2*r2*0.041666667f;
        output[idx] = poly * __int_as_float((k + 127) << 23);
    }
}

// Degree 5: exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120
__global__ void exp_poly5(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        const float inv_ln2 = 1.44269504f, ln2 = 0.69314718f;
        x = fmaxf(fminf(x, 88.0f), -88.0f);
        int k = __float2int_rn(x * inv_ln2);
        float r = x - k * ln2;
        float r2 = r * r;
        float poly = 1.0f + r + r2*0.5f + r2*r*0.16666667f
                   + r2*r2*0.041666667f + r2*r2*r*0.0083333333f;
        output[idx] = poly * __int_as_float((k + 127) << 23);
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// CPU reference computation
void exp_cpu(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = expf(input[i]);
    }
}

// Compute relative error
float compute_error(const float *approx, const float *reference, int n) {
    float max_rel_error = 0.0f;
    float max_abs_error = 0.0f;
    float sum_sq_rel_error = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float ref = reference[i];
        float app = approx[i];
        
        if (ref != 0.0f) {
            float rel_error = fabsf((app - ref) / ref);
            max_rel_error = fmaxf(max_rel_error, rel_error);
            sum_sq_rel_error += rel_error * rel_error;
        }
        
        float abs_error = fabsf(app - ref);
        max_abs_error = fmaxf(max_abs_error, abs_error);
    }
    
    // Return RMSE (root mean square error)
    return sqrtf(sum_sq_rel_error / n);
}

// ============================================================================
// BENCHMARK FUNCTION
// ============================================================================

struct BenchmarkResult {
    const char *name;
    float time_ms;
    float rmse_error;
    float max_error;
    float speedup;
};

BenchmarkResult benchmark_kernel(
    void (*kernel)(const float *, float *, int),
    const float *d_input,
    float *d_output,
    const float *h_reference,
    int n,
    int blocks,
    int threads_per_block,
    float baseline_time,
    const char *name
) {
    // Warm-up
    kernel<<<blocks, threads_per_block>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<blocks, threads_per_block>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    time_ms /= iterations;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Copy result back
    float *h_output = (float *)malloc(n * sizeof(float));
    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute error
    float rmse = compute_error(h_output, h_reference, n);
    
    // Compute max error for first 10 elements (for debugging)
    float max_error = 0.0f;
    for (int i = 0; i < std::min(n, 10); i++) {
        max_error = fmaxf(max_error, fabsf(h_output[i] - h_reference[i]));
    }
    
    free(h_output);
    
    BenchmarkResult result;
    result.name = name;
    result.time_ms = time_ms;
    result.rmse_error = rmse;
    result.max_error = max_error;
    result.speedup = baseline_time / time_ms;
    
    return result;
}

// ============================================================================
// MAIN BENCHMARK
// ============================================================================

int main() {
    int n = 1024 * 1024;  // 1M elements
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    printf("CUDA Exponential Approximation Benchmark\n");
    printf("=========================================\n");
    printf("Array size: %d elements\n", n);
    printf("Blocks: %d, Threads per block: %d\n\n", blocks, threads_per_block);
    
    // Allocate host memory
    float *h_input = (float *)malloc(n * sizeof(float));
    float *h_reference = (float *)malloc(n * sizeof(float));
    
    // Initialize input with small values (-2 to 0) typical for attention
    // This is important: large exponentials overflow, attention uses softmax normalization
    for (int i = 0; i < n; i++) {
        h_input[i] = -2.0f + (4.0f * rand() / RAND_MAX);  // Range [-2, 2]
    }
    
    // Compute CPU reference
    exp_cpu(h_input, h_reference, n);
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Benchmark hardware exponential (baseline)
    BenchmarkResult hw_result = benchmark_kernel(
        exp_hardware, d_input, d_output, h_reference, n, blocks, threads_per_block,
        1.0f, "Hardware __expf"
    );
    
    printf("Hardware Exponential (__expf):\n");
    printf("  Time: %.4f ms\n", hw_result.time_ms);
    printf("  RMSE: %.2e\n", hw_result.rmse_error);
    printf("  Max Error (first 10): %.2e\n\n", hw_result.max_error);
    
    // Benchmark polynomial approximations
    BenchmarkResult poly3_result = benchmark_kernel(
        exp_poly3, d_input, d_output, h_reference, n, blocks, threads_per_block,
        hw_result.time_ms, "Polynomial Degree 3"
    );
    
    printf("Polynomial Approximation (Degree 3):\n");
    printf("  Time: %.4f ms\n", poly3_result.time_ms);
    printf("  Speedup vs Hardware: %.2f x\n", poly3_result.speedup);
    printf("  RMSE: %.2e\n", poly3_result.rmse_error);
    printf("  Max Error (first 10): %.2e\n\n", poly3_result.max_error);
    
    BenchmarkResult poly4_result = benchmark_kernel(
        exp_poly4, d_input, d_output, h_reference, n, blocks, threads_per_block,
        hw_result.time_ms, "Polynomial Degree 4"
    );
    
    printf("Polynomial Approximation (Degree 4):\n");
    printf("  Time: %.4f ms\n", poly4_result.time_ms);
    printf("  Speedup vs Hardware: %.2f x\n", poly4_result.speedup);
    printf("  RMSE: %.2e\n", poly4_result.rmse_error);
    printf("  Max Error (first 10): %.2e\n\n", poly4_result.max_error);
    
    BenchmarkResult poly5_result = benchmark_kernel(
        exp_poly5, d_input, d_output, h_reference, n, blocks, threads_per_block,
        hw_result.time_ms, "Polynomial Degree 5"
    );
    
    printf("Polynomial Approximation (Degree 5):\n");
    printf("  Time: %.4f ms\n", poly5_result.time_ms);
    printf("  Speedup vs Hardware: %.2f x\n", poly5_result.speedup);
    printf("  RMSE: %.2e\n", poly5_result.rmse_error);
    printf("  Max Error (first 10): %.2e\n\n", poly5_result.max_error);
    
    // Write CSV for plotting
    FILE *csv_file = fopen("benchmark_results.csv", "w");
    fprintf(csv_file, "method,time_ms,rmse_error,speedup\n");
    fprintf(csv_file, "%s,%.4f,%.2e,%.4f\n", hw_result.name, hw_result.time_ms, hw_result.rmse_error, 1.0f);
    fprintf(csv_file, "%s,%.4f,%.2e,%.4f\n", poly3_result.name, poly3_result.time_ms, poly3_result.rmse_error, poly3_result.speedup);
    fprintf(csv_file, "%s,%.4f,%.2e,%.4f\n", poly4_result.name, poly4_result.time_ms, poly4_result.rmse_error, poly4_result.speedup);
    fprintf(csv_file, "%s,%.4f,%.2e,%.4f\n", poly5_result.name, poly5_result.time_ms, poly5_result.rmse_error, poly5_result.speedup);
    fclose(csv_file);
    
    printf("Results written to benchmark_results.csv\n");
    
    // Cleanup
    free(h_input);
    free(h_reference);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}
