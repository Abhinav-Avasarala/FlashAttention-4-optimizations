# Exponential Approximation in CUDA: FA-4 Benchmark Report

**Project:** CUDA Exponential Approximation Benchmark  
**Date:** [Fill in date]  
**Author:** [Fill in your name]  
**Institution:** NCSU  

---

## Executive Summary

This project implements and benchmarks polynomial exponential approximations in CUDA as used in Flash Attention 4 (FA-4). We compare hardware exponential (`__expf`) against degree-3, 4, and 5 polynomial approximations on real GPU hardware.

### Key Findings

- **Speedup Range:** [Fill in from results]
- **Accuracy Loss:** [Fill in from results]  
- **Recommended Method:** [Fill in recommendation]
- **GPU Used:** [Fill in your GPU model and specs]

---

## 1. Background: Why Polynomial Exponentials?

### 1.1 The Problem

In transformer models, the softmax operation is a bottleneck:

```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

The exponential function is **expensive** in terms of:
- **Latency**: Hardware exponential has high latency (~4-10 cycles on modern GPUs)
- **Register pressure**: `__expf` may require intermediate registers
- **Memory bandwidth**: Exponential-heavy workloads can become compute-bound

### 1.2 Polynomial Approximation Strategy

Instead of `exp(x)`, use Taylor series:

```
exp(x) ≈ 1 + x + x²/2! + x³/3! + x⁴/4! + x⁵/5! + ...
```

For small values (typical in attention: x ∈ [-5, 5]):
- **Degree 3**: 4 operations, ~1.7% error
- **Degree 4**: 5 operations, ~0.2% error
- **Degree 5**: 6 operations, ~0.01% error

### 1.3 Why This Matters

Flash Attention computes exponentials in a **tight loop** during online softmax:

```cuda
// Pseudo-code (simplified FA-4)
for (int i = 0; i < head_dim; i++) {
    float qk = q[i] * k[i];
    float exp_qk = __expf(qk - max_qk);  // ← BOTTLENECK
    sum_exp += exp_qk;
    // ... accumulate ...
}
```

Replacing `__expf` with a polynomial can reduce per-iteration latency.

---

## 2. Implementation Details

### 2.1 Kernels Implemented

#### Hardware Exponential (Baseline)
```cuda
__global__ void exp_hardware(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __expf(input[idx]);  // Native CUDA exponential
    }
}
```

**Characteristics:**
- 1 operation per element
- High accuracy (~ULP-accurate)
- Higher latency

#### Polynomial Degree 4
```cuda
__global__ void exp_poly4(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float x2 = x * x;
        float result = 1.0f + x + x2 * 0.5f + (x2 * x) * 0.16666667f + 
                       (x2 * x2) * 0.041666667f;
        output[idx] = result;
    }
}
```

**Characteristics:**
- 5 FMA (fused multiply-add) operations
- Lower latency than `__expf`
- Approximation error grows outside [-2, 2]

### 2.2 Test Data

- **Input range:** [-2, 2] (typical for attention after Q·K computation)
- **Array size:** 1M elements (realistic for batch processing)
- **Block configuration:** 256 threads/block

**Rationale:**
- Attention mechanisms usually normalize by max before exponential
- Max value is typically within [-5, 5] after normalization
- Working with [-2, 2] is where polynomials are most effective

### 2.3 Accuracy Metrics

We measure:

1. **RMSE (Root Mean Square Error)**
   ```
   RMSE = sqrt(1/n Σ(approx_i - reference_i)²)
   ```
   Good for understanding average deviation.

2. **KL Divergence (for softmax)**
   ```
   KL(p || q) = Σ p_i * log(p_i / q_i)
   ```
   Measures distribution divergence - important for softmax outputs.

3. **Top-1 Accuracy**
   Percentage of cases where polynomial and hardware select the same maximum element.

---

## 3. Experimental Setup

### 3.1 Hardware Configuration

**GPU:** [Fill in your GPU]
```
nvidia-smi output:
[Paste GPU information]
```

**Compute Capability:** [Fill in, e.g., sm_70, sm_80]  
**Memory:** [Fill in]  
**Peak FLOPs:** [Fill in]  

### 3.2 Compilation

```bash
# Load CUDA module (NCSU HPC specific)
module load cuda/[version]

# Compile with optimization flags
nvcc -O3 -arch=sm_70 exponential_kernel.cu -o exponential_kernel
```

### 3.3 Execution

```bash
# Run on GPU node
./exponential_kernel

# Monitor GPU usage
watch nvidia-smi
```

---

## 4. Results

### 4.1 Exponential Benchmark

**Table 1: Raw Benchmark Results**

| Method | Time (ms) | RMSE Error | Speedup |
|--------|-----------|------------|---------|
| Hardware __expf | [FILL] | [FILL] | 1.00x |
| Poly Degree 3 | [FILL] | [FILL] | [FILL]x |
| Poly Degree 4 | [FILL] | [FILL] | [FILL]x |
| Poly Degree 5 | [FILL] | [FILL] | [FILL]x |

### 4.2 Softmax Accuracy Test

**Table 2: Softmax Results (10K vocabulary)**

| Metric | Value |
|--------|-------|
| Hardware softmax time | [FILL] ms |
| Polynomial softmax time | [FILL] ms |
| Softmax speedup | [FILL]x |
| Average KL divergence | [FILL] |
| Top-1 match rate | [FILL]% |

### 4.3 Analysis

#### Speedup Analysis

[Fill in:]
- Which method achieved best speedup?
- Why? (Register pressure? Branch divergence?)
- Trade-off vs accuracy

#### Accuracy Analysis

[Fill in:]
- Which polynomial degree gives best accuracy?
- How does error grow outside [-2, 2]?
- Is top-1 prediction stable?

#### Practical Impact

[Fill in:]
- For softmax: is KL divergence acceptable?
- Would this affect training convergence?
- What about inference accuracy?

---

## 5. Findings & Discussion

### 5.1 Register Pressure

**Observation:** [Fill in]

The NVIDIA profiler shows:

```
nvprof --metrics all ./exponential_kernel
```

[Fill in profiler output highlights]

**Interpretation:**
- `__expf` has [?] register usage
- Polynomial has [?] register usage
- Occupancy impact: [?]

### 5.2 Instruction Throughput

Hardware exponential:
- Is a **specialized instruction** with long latency (~4-10 cycles)
- But high throughput on modern GPUs (pipelined)

Polynomial:
- Uses FMA operations (1 cycle latency, fully pipelined)
- Better instruction-level parallelism

### 5.3 Softmax Context

When used in softmax computation:

```cuda
for each i in sequence:
    exp_val = __expf(x[i] - max_val)  // Dependent chain!
    sum_exp += exp_val;                // Can hide latency?
```

- Is the dependency chain a bottleneck?
- Can polynomial reduce latency enough to matter?

### 5.4 Memory Hierarchy

- All data fits in L2 cache (1M floats = 4MB)
- No memory bandwidth issues
- Compute-bound workload

---

## 6. Conclusions

### 6.1 Summary

[Fill in your main conclusions]

### 6.2 Trade-offs

**Polynomial Approximation:**
- ✓ Faster
- ✓ Lower register pressure
- ✗ Accuracy loss
- ✗ Limited range

**Hardware Exponential:**
- ✓ High accuracy
- ✓ Unlimited range
- ✗ Slower
- ✗ Higher register pressure

### 6.3 Recommendations

[Fill in:]
- When to use polynomial approximation?
- When to stick with hardware exponential?
- Hybrid approaches?

**For Flash Attention 4:**
- Polynomial appears suitable for [-2, 2] range
- Would need validation on actual attention computation
- Consider mixed precision for improved accuracy

---

## 7. Appendix

### 7.1 Building and Running

```bash
# Clone/download code
cd cuda_exponential_benchmark

# Build
make all

# Run exponential benchmark
./bin/exponential_kernel

# Run softmax accuracy test
./bin/softmax_accuracy_test

# Generate plots
python3 plot_results.py
```

### 7.2 NCSU HPC Notes

**Loading CUDA:**
```bash
# Check available versions
module avail cuda

# Load specific version (adjust version)
module load cuda/12.0

# Verify
nvcc --version
```

**Job Submission (SLURM):**
```bash
#!/bin/bash
#SBATCH --job-name=exp_bench
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

module load cuda/12.0
cd /path/to/project
make run
```

### 7.3 GPU Architecture Notes

**V100 (sm_70):**
- 5120 CUDA cores
- 16GB memory
- High bandwidth (900 GB/s)

**A100 (sm_80):**
- 6912 CUDA cores
- 40GB memory
- Very high bandwidth (1.5 TB/s)

Adjust `-arch=sm_XX` flag in Makefile accordingly.

### 7.4 Further Work

Possible extensions:

1. **Range clipping:** What if we clip input to [-2, 2]?
2. **Hybrid approach:** Use polynomial for small values, hardware for large
3. **Different precisions:** Test with float16 or bfloat16
4. **Attention simulation:** Full transformer block with polynomial exponential
5. **Training stability:** Does polynomial affect convergence?

---

## References

1. Dao et al., "Flash-2: Faster Large Language Model Training with Scaled RoPE Attention" (2024)
2. NVIDIA CUDA C++ Programming Guide
3. Microarchitecture documentation for your GPU

---

**Last Updated:** [Date]
