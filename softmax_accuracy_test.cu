#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

// ============================================================================
// SOFTMAX WITH HARDWARE EXPONENTIAL
// ============================================================================

__global__ void softmax_hardware(const float *logits, float *output, int batch_size, int vocab_size) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float *batch_logits = logits + batch_idx * vocab_size;
    float *batch_output = output + batch_idx * vocab_size;
    
    // Find max (numerical stability)
    __shared__ float max_val;
    __shared__ float sum_exp;
    
    if (tid == 0) {
        max_val = batch_logits[0];
        for (int i = 1; i < vocab_size; i++) {
            max_val = fmaxf(max_val, batch_logits[i]);
        }
        sum_exp = 0.0f;
    }
    __syncthreads();
    
    // Compute exp and sum
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        batch_output[i] = __expf(batch_logits[i] - max_val);
        atomicAdd(&sum_exp, batch_output[i]);
    }
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        batch_output[i] /= sum_exp;
    }
}

// ============================================================================
// SOFTMAX WITH POLYNOMIAL APPROXIMATION (DEGREE 4)
// ============================================================================

__global__ void softmax_poly4(const float *logits, float *output, int batch_size, int vocab_size) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float *batch_logits = logits + batch_idx * vocab_size;
    float *batch_output = output + batch_idx * vocab_size;
    
    // Find max (numerical stability)
    __shared__ float max_val;
    __shared__ float sum_exp;
    
    if (tid == 0) {
        max_val = batch_logits[0];
        for (int i = 1; i < vocab_size; i++) {
            max_val = fmaxf(max_val, batch_logits[i]);
        }
        sum_exp = 0.0f;
    }
    __syncthreads();
    
    // Compute exp (polynomial with range reduction) and sum
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float x = batch_logits[i] - max_val;  // x <= 0 always
        // Range reduction: exp(x) = 2^k * exp(r), |r| <= ln2/2
        const float inv_ln2 = 1.44269504f, ln2 = 0.69314718f;
        x = fmaxf(x, -88.0f);
        int k = __float2int_rn(x * inv_ln2);
        float r = x - k * ln2;
        float r2 = r * r;
        float poly = 1.0f + r + r2*0.5f + r2*r*0.16666667f + r2*r2*0.041666667f;
        batch_output[i] = poly * __int_as_float((k + 127) << 23);
        atomicAdd(&sum_exp, batch_output[i]);
    }
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        batch_output[i] /= sum_exp;
    }
}

// ============================================================================
// SOFTMAX ACCURACY TEST
// ============================================================================

float compute_softmax_kl_divergence(const float *approx, const float *reference, int vocab_size) {
    float kl_div = 0.0f;
    
    for (int i = 0; i < vocab_size; i++) {
        float p = reference[i];
        float q = approx[i];
        
        // Avoid log(0)
        if (p > 1e-7 && q > 1e-7) {
            kl_div += p * logf(p / q);
        }
    }
    
    return kl_div;
}

float compute_softmax_max_diff(const float *approx, const float *reference, int vocab_size) {
    float max_diff = 0.0f;
    
    for (int i = 0; i < vocab_size; i++) {
        float diff = fabsf(approx[i] - reference[i]);
        max_diff = fmaxf(max_diff, diff);
    }
    
    return max_diff;
}

int main() {
    int batch_size = 32;
    int vocab_size = 10000;  // Typical transformer vocab
    int total_elements = batch_size * vocab_size;
    
    printf("Softmax Accuracy Test (Attention-like scenario)\n");
    printf("==============================================\n");
    printf("Batch size: %d\n", batch_size);
    printf("Vocab size: %d\n", vocab_size);
    printf("Total elements: %d\n\n", total_elements);
    
    // Allocate host memory
    float *h_logits = (float *)malloc(total_elements * sizeof(float));
    float *h_softmax_hw = (float *)malloc(total_elements * sizeof(float));
    float *h_softmax_poly = (float *)malloc(total_elements * sizeof(float));
    
    // Initialize logits (typical attention scores before softmax)
    // Use smaller range for better polynomial approximation
    for (int i = 0; i < total_elements; i++) {
        h_logits[i] = -5.0f + (10.0f * rand() / RAND_MAX);  // Range [-5, 5]
    }
    
    // Allocate device memory
    float *d_logits, *d_output_hw, *d_output_poly;
    cudaMalloc(&d_logits, total_elements * sizeof(float));
    cudaMalloc(&d_output_hw, total_elements * sizeof(float));
    cudaMalloc(&d_output_poly, total_elements * sizeof(float));
    
    // Copy logits to device
    cudaMemcpy(d_logits, h_logits, total_elements * sizeof(float), cudaMemcpyHostToDevice);
    
    // Warm-up
    softmax_hardware<<<batch_size, 256>>>(d_logits, d_output_hw, batch_size, vocab_size);
    cudaDeviceSynchronize();
    
    // Benchmark hardware softmax
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        softmax_hardware<<<batch_size, 256>>>(d_logits, d_output_hw, batch_size, vocab_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_hw_ms;
    cudaEventElapsedTime(&time_hw_ms, start, stop);
    time_hw_ms /= iterations;
    
    // Benchmark polynomial softmax
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        softmax_poly4<<<batch_size, 256>>>(d_logits, d_output_poly, batch_size, vocab_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_poly_ms;
    cudaEventElapsedTime(&time_poly_ms, start, stop);
    time_poly_ms /= iterations;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Copy results back
    cudaMemcpy(h_softmax_hw, d_output_hw, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_softmax_poly, d_output_poly, total_elements * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute accuracy metrics
    printf("Softmax Accuracy Comparison:\n");
    printf("-----------------------------\n\n");
    
    float total_kl_div = 0.0f;
    float total_max_diff = 0.0f;
    float max_kl_div = 0.0f;
    float max_max_diff = 0.0f;
    
    for (int batch = 0; batch < batch_size; batch++) {
        int offset = batch * vocab_size;
        float kl_div = compute_softmax_kl_divergence(
            h_softmax_poly + offset,
            h_softmax_hw + offset,
            vocab_size
        );
        float max_diff = compute_softmax_max_diff(
            h_softmax_poly + offset,
            h_softmax_hw + offset,
            vocab_size
        );
        
        total_kl_div += kl_div;
        total_max_diff += max_diff;
        max_kl_div = fmaxf(max_kl_div, kl_div);
        max_max_diff = fmaxf(max_max_diff, max_diff);
    }
    
    float avg_kl_div = total_kl_div / batch_size;
    float avg_max_diff = total_max_diff / batch_size;
    
    printf("Hardware Softmax Time: %.4f ms\n", time_hw_ms);
    printf("Polynomial Softmax Time: %.4f ms\n", time_poly_ms);
    printf("Speedup: %.2f x\n\n", time_hw_ms / time_poly_ms);
    
    printf("Accuracy Metrics (Polynomial vs Hardware):\n");
    printf("  Average KL Divergence: %.2e\n", avg_kl_div);
    printf("  Max KL Divergence: %.2e\n", max_kl_div);
    printf("  Average Max Difference: %.2e\n", avg_max_diff);
    printf("  Max Difference: %.2e\n\n", max_max_diff);
    
    // Check if top-1 prediction is the same
    int top1_matches = 0;
    for (int batch = 0; batch < batch_size; batch++) {
        int offset = batch * vocab_size;
        
        // Find top-1 for hardware
        int top1_hw = 0;
        float max_hw = h_softmax_hw[offset];
        for (int i = 1; i < vocab_size; i++) {
            if (h_softmax_hw[offset + i] > max_hw) {
                max_hw = h_softmax_hw[offset + i];
                top1_hw = i;
            }
        }
        
        // Find top-1 for polynomial
        int top1_poly = 0;
        float max_poly = h_softmax_poly[offset];
        for (int i = 1; i < vocab_size; i++) {
            if (h_softmax_poly[offset + i] > max_poly) {
                max_poly = h_softmax_poly[offset + i];
                top1_poly = i;
            }
        }
        
        if (top1_hw == top1_poly) {
            top1_matches++;
        }
    }
    
    printf("Top-1 Prediction Matching: %d/%d (%.1f%%)\n\n",
           top1_matches, batch_size, 100.0f * top1_matches / batch_size);
    
    // Write results to CSV
    FILE *csv_file = fopen("softmax_accuracy_results.csv", "w");
    fprintf(csv_file, "metric,value\n");
    fprintf(csv_file, "avg_kl_divergence,%.2e\n", avg_kl_div);
    fprintf(csv_file, "max_kl_divergence,%.2e\n", max_kl_div);
    fprintf(csv_file, "avg_max_diff,%.2e\n", avg_max_diff);
    fprintf(csv_file, "hardware_time_ms,%.4f\n", time_hw_ms);
    fprintf(csv_file, "poly_time_ms,%.4f\n", time_poly_ms);
    fprintf(csv_file, "speedup,%.2f\n", time_hw_ms / time_poly_ms);
    fprintf(csv_file, "top1_match_rate,%.2f\n", 100.0f * top1_matches / batch_size);
    fclose(csv_file);
    
    printf("Results written to softmax_accuracy_results.csv\n");
    
    // Cleanup
    free(h_logits);
    free(h_softmax_hw);
    free(h_softmax_poly);
    cudaFree(d_logits);
    cudaFree(d_output_hw);
    cudaFree(d_output_poly);
    
    return 0;
}
