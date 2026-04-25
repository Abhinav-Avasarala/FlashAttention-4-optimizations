#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

#define ITERS 512
#define SCALE 9.54e-7f  // feedback scale: keeps x in [-1,1] to prevent overflow

// ============================================================================
// ACCURACY KERNELS — single pass, used to measure RMSE vs CPU reference
// ============================================================================

__global__ void exp_hw_acc(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __expf(in[i]);
}

__global__ void exp_poly3_acc(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    const float inv_ln2 = 1.44269504f, ln2 = 0.69314718f;
    x = fmaxf(fminf(x, 88.0f), -88.0f);
    int k = __float2int_rn(x * inv_ln2);
    float r = x - k * ln2;
    float poly = 1.0f + r + r*r*0.5f + r*r*r*0.16666667f;
    out[i] = poly * __int_as_float((k + 127) << 23);
}

__global__ void exp_poly4_acc(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    const float inv_ln2 = 1.44269504f, ln2 = 0.69314718f;
    x = fmaxf(fminf(x, 88.0f), -88.0f);
    int k = __float2int_rn(x * inv_ln2);
    float r = x - k * ln2, r2 = r*r;
    float poly = 1.0f + r + r2*0.5f + r2*r*0.16666667f + r2*r2*0.041666667f;
    out[i] = poly * __int_as_float((k + 127) << 23);
}

__global__ void exp_poly5_acc(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    const float inv_ln2 = 1.44269504f, ln2 = 0.69314718f;
    x = fmaxf(fminf(x, 88.0f), -88.0f);
    int k = __float2int_rn(x * inv_ln2);
    float r = x - k * ln2, r2 = r*r;
    float poly = 1.0f + r + r2*0.5f + r2*r*0.16666667f
               + r2*r2*0.041666667f + r2*r2*r*0.0083333333f;
    out[i] = poly * __int_as_float((k + 127) << 23);
}

// ============================================================================
// COMPUTE-BOUND TIMING KERNELS — ITERS exp calls per thread, register-resident
// Feedback loop (x = acc * SCALE) keeps x in range and prevents dead-code
// elimination, mirroring how exp() is used inside a fused attention kernel.
// ============================================================================

__global__ void exp_hw_cb(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i], acc = 0.0f;
    #pragma unroll 8
    for (int j = 0; j < ITERS; j++) { acc += __expf(x); x = acc * SCALE; }
    out[i] = acc;
}

__global__ void exp_poly3_cb(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i], acc = 0.0f;
    #pragma unroll 8
    for (int j = 0; j < ITERS; j++) {
        const float inv_ln2 = 1.44269504f, ln2 = 0.69314718f;
        float xc = fmaxf(fminf(x, 88.0f), -88.0f);
        int k = __float2int_rn(xc * inv_ln2);
        float r = xc - k * ln2;
        acc += (1.0f + r + r*r*0.5f + r*r*r*0.16666667f) * __int_as_float((k+127)<<23);
        x = acc * SCALE;
    }
    out[i] = acc;
}

__global__ void exp_poly4_cb(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i], acc = 0.0f;
    #pragma unroll 8
    for (int j = 0; j < ITERS; j++) {
        const float inv_ln2 = 1.44269504f, ln2 = 0.69314718f;
        float xc = fmaxf(fminf(x, 88.0f), -88.0f);
        int k = __float2int_rn(xc * inv_ln2);
        float r = xc - k * ln2, r2 = r*r;
        acc += (1.0f + r + r2*0.5f + r2*r*0.16666667f + r2*r2*0.041666667f)
               * __int_as_float((k+127)<<23);
        x = acc * SCALE;
    }
    out[i] = acc;
}

__global__ void exp_poly5_cb(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i], acc = 0.0f;
    #pragma unroll 8
    for (int j = 0; j < ITERS; j++) {
        const float inv_ln2 = 1.44269504f, ln2 = 0.69314718f;
        float xc = fmaxf(fminf(x, 88.0f), -88.0f);
        int k = __float2int_rn(xc * inv_ln2);
        float r = xc - k * ln2, r2 = r*r;
        acc += (1.0f + r + r2*0.5f + r2*r*0.16666667f
                + r2*r2*0.041666667f + r2*r2*r*0.0083333333f)
               * __int_as_float((k+127)<<23);
        x = acc * SCALE;
    }
    out[i] = acc;
}

// ============================================================================
// HELPERS
// ============================================================================

void exp_cpu(const float *in, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = expf(in[i]);
}

float compute_rmse(const float *approx, const float *ref, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        if (ref[i] != 0.0f) {
            float e = (approx[i] - ref[i]) / ref[i];
            sum += e * e;
        }
    }
    return sqrtf(sum / n);
}

float time_kernel(void (*kernel)(const float*, float*, int),
                  const float *d_in, float *d_out, int n, int blocks, int threads) {
    // warmup
    for (int i = 0; i < 3; i++) kernel<<<blocks, threads>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) kernel<<<blocks, threads>>>(d_in, d_out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / 20;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    int n = 1024 * 1024, threads = 256;
    int blocks = (n + threads - 1) / threads;

    printf("CUDA Exponential Approximation Benchmark (Compute-Bound)\n");
    printf("==========================================================\n");
    printf("Elements: %d  |  Exp calls per thread (timing): %d\n\n", n, ITERS);

    float *h_in = (float*)malloc(n * sizeof(float));
    float *h_ref = (float*)malloc(n * sizeof(float));
    float *h_out = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h_in[i] = -2.0f + 4.0f * rand() / RAND_MAX;
    exp_cpu(h_in, h_ref, n);

    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    // ── Measure RMSE from single-pass accuracy kernels ──
    void (*acc_kernels[4])(const float*, float*, int) = {
        exp_hw_acc, exp_poly3_acc, exp_poly4_acc, exp_poly5_acc
    };
    float rmse[4];
    for (int k = 0; k < 4; k++) {
        acc_kernels[k]<<<blocks, threads>>>(d_in, d_out, n);
        cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        rmse[k] = compute_rmse(h_out, h_ref, n);
    }

    // ── Measure timing from compute-bound kernels ──
    void (*cb_kernels[4])(const float*, float*, int) = {
        exp_hw_cb, exp_poly3_cb, exp_poly4_cb, exp_poly5_cb
    };
    const char *names[4] = {
        "Hardware __expf", "Polynomial Degree 3", "Polynomial Degree 4", "Polynomial Degree 5"
    };
    float times[4];
    for (int k = 0; k < 4; k++)
        times[k] = time_kernel(cb_kernels[k], d_in, d_out, n, blocks, threads);

    // ── Print results ──
    printf("%-22s  %10s  %12s  %8s\n", "Method", "Time (ms)", "RMSE", "Speedup");
    printf("%-22s  %10s  %12s  %8s\n", "------", "---------", "----", "-------");
    for (int k = 0; k < 4; k++) {
        printf("%-22s  %10.4f  %12.2e  %8.2fx\n",
               names[k], times[k], rmse[k], times[0] / times[k]);
    }

    // ── Write CSV ──
    FILE *f = fopen("benchmark_results.csv", "w");
    fprintf(f, "method,time_ms,rmse_error,speedup\n");
    for (int k = 0; k < 4; k++)
        fprintf(f, "%s,%.4f,%.2e,%.4f\n", names[k], times[k], rmse[k], times[0]/times[k]);
    fclose(f);
    printf("\nResults written to benchmark_results.csv\n");

    free(h_in); free(h_ref); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
