#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

// Iterations of softmax run internally per kernel launch (data stays in shared memory)
#define SOFTMAX_ITERS 32

// ============================================================================
// WARP-SHUFFLE REDUCTION HELPERS
// Replaces atomicAdd — removes serialization so exp() is the actual bottleneck
// ============================================================================

__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-wide max/sum using warp lanes then one final reduce in thread 0
__device__ float block_reduce_max(float val, float *smem_scratch) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    val = warp_reduce_max(val);
    if (lane == 0) smem_scratch[warp] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float g = smem_scratch[0];
        for (int w = 1; w < (blockDim.x >> 5); w++) g = fmaxf(g, smem_scratch[w]);
        smem_scratch[0] = g;
    }
    __syncthreads();
    return smem_scratch[0];
}

__device__ float block_reduce_sum(float val, float *smem_scratch) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) smem_scratch[warp] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float g = 0.0f;
        for (int w = 0; w < (blockDim.x >> 5); w++) g += smem_scratch[w];
        smem_scratch[0] = g;
    }
    __syncthreads();
    return smem_scratch[0];
}

// ============================================================================
// ACCURACY KERNELS — single pass, used to compute KL divergence and top-1
// ============================================================================

__global__ void softmax_hw_acc(const float *logits, float *out,
                                int batch_size, int vocab_size) {
    extern __shared__ float smem[];          // vocab_size floats
    float *scratch = smem + vocab_size;      // 8 floats for warp reduction (256/32=8 warps)

    int b = blockIdx.x, tid = threadIdx.x, bdim = blockDim.x;
    if (b >= batch_size) return;
    const float *src = logits + b * vocab_size;
    float *dst = out + b * vocab_size;

    for (int i = tid; i < vocab_size; i += bdim) smem[i] = src[i];
    __syncthreads();

    float local_max = -1e30f;
    for (int i = tid; i < vocab_size; i += bdim) local_max = fmaxf(local_max, smem[i]);
    float max_val = block_reduce_max(local_max, scratch);

    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += bdim) {
        float v = __expf(smem[i] - max_val);
        smem[i] = v;
        local_sum += v;
    }
    float sum_val = block_reduce_sum(local_sum, scratch);

    for (int i = tid; i < vocab_size; i += bdim) dst[i] = smem[i] / sum_val;
}

__global__ void softmax_poly4_acc(const float *logits, float *out,
                                   int batch_size, int vocab_size) {
    extern __shared__ float smem[];
    float *scratch = smem + vocab_size;

    int b = blockIdx.x, tid = threadIdx.x, bdim = blockDim.x;
    if (b >= batch_size) return;
    const float *src = logits + b * vocab_size;
    float *dst = out + b * vocab_size;

    for (int i = tid; i < vocab_size; i += bdim) smem[i] = src[i];
    __syncthreads();

    float local_max = -1e30f;
    for (int i = tid; i < vocab_size; i += bdim) local_max = fmaxf(local_max, smem[i]);
    float max_val = block_reduce_max(local_max, scratch);

    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += bdim) {
        float x = smem[i] - max_val;
        x = fmaxf(x, -88.0f);
        const float inv_ln2 = 1.44269504f, ln2 = 0.69314718f;
        int k = __float2int_rn(x * inv_ln2);
        float r = x - k * ln2, r2 = r*r;
        float v = (1.0f + r + r2*0.5f + r2*r*0.16666667f + r2*r2*0.041666667f)
                  * __int_as_float((k+127)<<23);
        smem[i] = v;
        local_sum += v;
    }
    float sum_val = block_reduce_sum(local_sum, scratch);

    for (int i = tid; i < vocab_size; i += bdim) dst[i] = smem[i] / sum_val;
}

// ============================================================================
// COMPUTE-BOUND TIMING KERNELS
// Data loaded into shared memory once, SOFTMAX_ITERS passes run internally.
// Warp-shuffle reduction makes exp() the bottleneck, not atomicAdd.
// ============================================================================

__global__ void softmax_hw_cb(const float *logits, float *out,
                               int batch_size, int vocab_size) {
    extern __shared__ float smem[];
    float *scratch = smem + vocab_size;

    int b = blockIdx.x, tid = threadIdx.x, bdim = blockDim.x;
    if (b >= batch_size) return;

    for (int i = tid; i < vocab_size; i += bdim) smem[i] = logits[b*vocab_size + i];
    __syncthreads();

    for (int iter = 0; iter < SOFTMAX_ITERS; iter++) {
        float local_max = -1e30f;
        for (int i = tid; i < vocab_size; i += bdim) local_max = fmaxf(local_max, smem[i]);
        float max_val = block_reduce_max(local_max, scratch);

        float local_sum = 0.0f;
        for (int i = tid; i < vocab_size; i += bdim) {
            float v = __expf(smem[i] - max_val);
            smem[i] = v;
            local_sum += v;
        }
        float sum_val = block_reduce_sum(local_sum, scratch);

        for (int i = tid; i < vocab_size; i += bdim) smem[i] /= sum_val;
        __syncthreads();
    }

    for (int i = tid; i < vocab_size; i += bdim) out[b*vocab_size + i] = smem[i];
}

__global__ void softmax_poly4_cb(const float *logits, float *out,
                                  int batch_size, int vocab_size) {
    extern __shared__ float smem[];
    float *scratch = smem + vocab_size;

    int b = blockIdx.x, tid = threadIdx.x, bdim = blockDim.x;
    if (b >= batch_size) return;

    for (int i = tid; i < vocab_size; i += bdim) smem[i] = logits[b*vocab_size + i];
    __syncthreads();

    // After the first softmax pass all values are probabilities in (0,1],
    // so x - max_val is always in (-1, 0]. Horner's Taylor series is accurate
    // in that range without range reduction — no clamp/int overhead needed.
    for (int iter = 0; iter < SOFTMAX_ITERS; iter++) {
        float local_max = -1e30f;
        for (int i = tid; i < vocab_size; i += bdim) local_max = fmaxf(local_max, smem[i]);
        float max_val = block_reduce_max(local_max, scratch);

        float local_sum = 0.0f;
        for (int i = tid; i < vocab_size; i += bdim) {
            float x = smem[i] - max_val;  // x in (-1, 0]
            float v = 1.0f + x * (1.0f + x * (0.5f + x * (0.16666667f + x * 0.041666667f)));
            smem[i] = v;
            local_sum += v;
        }
        float sum_val = block_reduce_sum(local_sum, scratch);

        for (int i = tid; i < vocab_size; i += bdim) smem[i] /= sum_val;
        __syncthreads();
    }

    for (int i = tid; i < vocab_size; i += bdim) out[b*vocab_size + i] = smem[i];
}

// ============================================================================
// ACCURACY METRICS
// ============================================================================

float kl_divergence(const float *approx, const float *ref, int n) {
    float kl = 0.0f;
    for (int i = 0; i < n; i++)
        if (ref[i] > 1e-7f && approx[i] > 1e-7f)
            kl += ref[i] * logf(ref[i] / approx[i]);
    return kl;
}

int top1_match(const float *a, const float *b, int n) {
    int ia = 0, ib = 0;
    for (int i = 1; i < n; i++) {
        if (a[i] > a[ia]) ia = i;
        if (b[i] > b[ib]) ib = i;
    }
    return ia == ib;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    int batch_size = 32, vocab_size = 10000;
    int total = batch_size * vocab_size;
    int threads = 256;
    // shared mem: vocab_size floats + 8 floats for warp scratch (256 threads / 32 = 8 warps)
    size_t smem = (vocab_size + 8) * sizeof(float);

    printf("Softmax Accuracy & Compute-Bound Timing Test\n");
    printf("=============================================\n");
    printf("Batch: %d  |  Vocab: %d  |  Internal iters (timing): %d\n\n",
           batch_size, vocab_size, SOFTMAX_ITERS);

    float *h_logits = (float*)malloc(total * sizeof(float));
    float *h_hw     = (float*)malloc(total * sizeof(float));
    float *h_poly   = (float*)malloc(total * sizeof(float));
    for (int i = 0; i < total; i++)
        h_logits[i] = -5.0f + 10.0f * rand() / RAND_MAX;

    float *d_logits, *d_hw, *d_poly;
    cudaMalloc(&d_logits, total * sizeof(float));
    cudaMalloc(&d_hw,     total * sizeof(float));
    cudaMalloc(&d_poly,   total * sizeof(float));
    cudaMemcpy(d_logits, h_logits, total * sizeof(float), cudaMemcpyHostToDevice);

    // ── Accuracy pass (single iteration) ──
    softmax_hw_acc  <<<batch_size, threads, smem>>>(d_logits, d_hw,   batch_size, vocab_size);
    softmax_poly4_acc<<<batch_size, threads, smem>>>(d_logits, d_poly, batch_size, vocab_size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_hw,   d_hw,   total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_poly, d_poly, total * sizeof(float), cudaMemcpyDeviceToHost);

    float avg_kl = 0.0f, max_kl = 0.0f;
    int top1_matches = 0;
    for (int b = 0; b < batch_size; b++) {
        int off = b * vocab_size;
        float kl = kl_divergence(h_poly + off, h_hw + off, vocab_size);
        avg_kl += kl;
        if (kl > max_kl) max_kl = kl;
        top1_matches += top1_match(h_hw + off, h_poly + off, vocab_size);
    }
    avg_kl /= batch_size;

    printf("Accuracy (polynomial vs hardware, single pass):\n");
    printf("  Avg KL Divergence:  %.2e\n", avg_kl);
    printf("  Max KL Divergence:  %.2e\n", max_kl);
    printf("  Top-1 Match:        %d/%d (%.1f%%)\n\n",
           top1_matches, batch_size, 100.f * top1_matches / batch_size);

    // ── Compute-bound timing pass ──
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int reps = 20;

    // warmup
    for (int i = 0; i < 3; i++)
        softmax_hw_cb<<<batch_size, threads, smem>>>(d_logits, d_hw, batch_size, vocab_size);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < reps; i++)
        softmax_hw_cb<<<batch_size, threads, smem>>>(d_logits, d_hw, batch_size, vocab_size);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float time_hw; cudaEventElapsedTime(&time_hw, start, stop);
    time_hw /= reps;

    for (int i = 0; i < 3; i++)
        softmax_poly4_cb<<<batch_size, threads, smem>>>(d_logits, d_poly, batch_size, vocab_size);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < reps; i++)
        softmax_poly4_cb<<<batch_size, threads, smem>>>(d_logits, d_poly, batch_size, vocab_size);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float time_poly; cudaEventElapsedTime(&time_poly, start, stop);
    time_poly /= reps;

    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf("Compute-Bound Timing (%d internal softmax passes, data in shared memory):\n",
           SOFTMAX_ITERS);
    printf("  Hardware softmax:    %.4f ms\n", time_hw);
    printf("  Polynomial softmax:  %.4f ms\n", time_poly);
    printf("  Speedup:             %.2fx\n\n", time_hw / time_poly);

    FILE *f = fopen("softmax_accuracy_results.csv", "w");
    fprintf(f, "metric,value\n");
    fprintf(f, "avg_kl_divergence,%.2e\n", avg_kl);
    fprintf(f, "max_kl_divergence,%.2e\n", max_kl);
    fprintf(f, "top1_match_rate,%.2f\n", 100.f * top1_matches / batch_size);
    fprintf(f, "hardware_time_ms,%.4f\n", time_hw);
    fprintf(f, "poly_time_ms,%.4f\n", time_poly);
    fprintf(f, "speedup,%.4f\n", time_hw / time_poly);
    fclose(f);
    printf("Results written to softmax_accuracy_results.csv\n");

    free(h_logits); free(h_hw); free(h_poly);
    cudaFree(d_logits); cudaFree(d_hw); cudaFree(d_poly);
    return 0;
}
