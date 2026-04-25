# CUDA Polynomial Exponential Approximation — Project Report

**Author:** Abhinav Avasarala | **Institution:** NCSU | **Date:** April 2026  
**GPU tested on:** NVIDIA H100 SXM (sm_90) — see `results.md` for full numbers

---

## 1. Motivation

Transformer models compute softmax in the attention mechanism:

```
softmax(x)_i = exp(x_i - max_x) / Σ_j exp(x_j - max_x)
```

The subtracting of `max_x` is a standard numerical stability trick — it ensures the largest input to `exp()` is always 0, so `exp(0) = 1` and all other values are in `(0, 1]`. This bounds the input range.

`exp()` is called for every element of every attention head at every layer. In a fused attention kernel (like Flash Attention), it is computed on data already in registers — making it potentially compute-bound. Flash Attention 4 proposes replacing `__expf` with a polynomial approximation to exploit this.

---

## 2. Mathematical Background

### 2.1 Taylor Series

The Taylor series expands `exp(x)` as an infinite sum around x = 0:

```
exp(x) = 1 + x + x²/2! + x³/3! + x⁴/4! + x⁵/5! + ...
       = Σ_{n=0}^{∞}  xⁿ / n!
```

Truncating at degree k gives a polynomial approximation. The error (remainder) is bounded by the next term:

```
|error| ≤ |x|^(k+1) / (k+1)!
```

This means accuracy degrades fast for large |x|, but for |x| ≤ 1 even a degree-3 polynomial is quite accurate.

### 2.2 The Three Polynomials

| Degree | Formula | Error bound for |x| ≤ 1 |
|--------|---------|------------------------|
| 3 | `1 + x + x²/2 + x³/6` | ≤ x⁴/24 ≈ 0.04 |
| 4 | `+ x⁴/24` | ≤ x⁵/120 ≈ 0.008 |
| 5 | `+ x⁵/120` | ≤ x⁶/720 ≈ 0.001 |

Each added term costs one more FMA (fused multiply-add) instruction but buys roughly 10x better accuracy.

### 2.3 Horner's Method

The naive expansion recomputes powers: `1 + x + x²*0.5 + x³*0.1667 + x⁴*0.0417`.  
Horner's method rewrites the same polynomial to minimize multiplications:

```
1 + x*(1 + x*(0.5 + x*(0.1667 + x*0.0417)))
```

Both evaluate to the same result, but Horner's form:
- Uses only N multiplications for degree N (vs 2N-1 naive)
- Has a strict sequential dependency chain (each step waits on the previous)
- Uses fewer intermediate registers

We use Horner's form in the compute-bound timing kernels.

### 2.4 Range Reduction

The Taylor series only converges accurately near x = 0. For arbitrary x, we first decompose:

```
exp(x) = 2^k * exp(r)
```

where k = round(x / ln2) is an integer and r = x - k·ln2 is the small remainder with |r| ≤ ln2/2 ≈ 0.347. We then:
1. Evaluate the polynomial on r (small, so series converges fast)
2. Multiply by 2^k using a bit trick: in IEEE 754, `(k + 127) << 23` placed in the float exponent field gives 2^k exactly

```cuda
int k = __float2int_rn(x * 1.44269504f);   // 1/ln2
float r = x - k * 0.69314718f;             // ln2
float poly = /* Horner on r */;
result = poly * __int_as_float((k + 127) << 23);
```

Range reduction is used in the **accuracy kernels** (single pass, arbitrary input). It is NOT used in the **timing kernels** because the feedback loop keeps x bounded.

---

## 3. Why Memory-Bound Kernels Show No Speedup

Our first implementation ran a single `exp()` per element — load from global memory, compute, store. This is **memory-bandwidth bound**: the GPU spends most of its time waiting for data, not computing.

```
1M floats × 2 (read + write) × 4 bytes = 8 MB
H100 bandwidth: ~3.35 TB/s
Minimum time: ~0.0024 ms
Measured time: ~0.005 ms  ← already near bandwidth limit
```

Replacing `__expf` with a polynomial saves a few nanoseconds of compute that was already hidden behind memory latency. Speedup: ~1.00x regardless of polynomial degree. This is not a bug — it is the expected behavior for a bandwidth-bound kernel.

---

## 4. Making It Compute-Bound

To isolate the exp() throughput, we restructure the kernel to do **512 exp() calls per thread entirely in registers** before writing once:

```cuda
float x = input[idx];
float acc = 0.0f;
#pragma unroll 8
for (int j = 0; j < 512; j++) {
    acc += __expf(x);          // or polynomial
    x = acc * 9.54e-7f;        // feedback: keeps x in [-1, 1]
}
output[idx] = acc;
```

The feedback `x = acc * SCALE` serves two purposes:
1. Prevents the compiler from treating the loop as dead code
2. Keeps x bounded in [-1, 1], so the Taylor series remains accurate without range reduction

Now the kernel is compute-bound — the GPU is spending all its cycles on exp() math, exactly as it would inside a fused Flash Attention kernel where exp() runs on register-resident QK^T scores.

### Softmax Compute-Bound

For softmax, data cannot stay in registers (10,000 elements per row). Instead, we load each row into **shared memory** once, then run 32 softmax passes internally. This keeps data in fast on-chip memory across iterations, making exp() the bottleneck.

We also replace `atomicAdd` (which serializes threads) with **warp-shuffle reduction**:

```cuda
// Each warp reduces its partial sum using shuffle:
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
```

This is fully parallel within each warp and removes the atomicAdd serialization, so exp() becomes the actual bottleneck.

---

## 5. Results Summary

All results on **NVIDIA H100 SXM (sm_90)**. See `results.md` for full tables.

### Exponential kernel (compute-bound, 512 calls/thread):

| Method | Speedup | RMSE |
|--------|---------|------|
| Polynomial Degree 3 | **1.52x** | 2.2e-4 |
| Polynomial Degree 4 | **1.29x** | 1.4e-5 |
| Polynomial Degree 5 | **1.12x** | 7.3e-7 |

### Softmax (compute-bound, shared memory):

| Method | Speedup | Top-1 Match |
|--------|---------|-------------|
| Polynomial Degree 4 | **1.15x** | 100% |

### A100 vs H100 comparison:

The speedup is **architecture-specific**. On A100 (sm_80), the polynomial showed no speedup in either benchmark. On H100 (sm_90), it wins clearly. This is because Hopper's FMA-to-ex2 throughput ratio is different from Ampere — the hardware `ex2` instruction is relatively more expensive on H100, making the FMA-based polynomial competitive.

This is why Flash Attention 4 explicitly targets H100.

---

## 6. Degree 4 as the Sweet Spot

Degree 4 is Flash Attention 4's recommended choice:

- **1.29x speedup** on H100 in compute-bound regime
- **RMSE = 1.4e-5** — accurate enough that softmax top-1 match stays 100%
- KL divergence ≈ 0 (8.4e-8) — distributions are statistically identical
- One more FMA than degree 3, but buys 16× better accuracy

Degree 3 is faster (1.52x) but the 2.2e-4 RMSE may affect training stability. Degree 5 buys almost nothing over degree 4 in accuracy while being slower.

---

## 7. Repository Structure

```
FlashAttention-4-optimizations/
├── exponential_kernel.cu       # Exp benchmark: accuracy (_acc) + compute-bound (_cb) kernels
├── softmax_accuracy_test.cu    # Softmax: accuracy pass + shared-memory compute-bound timing
├── plot_results.py             # Generates accuracy_comparison.png, detailed_analysis.png
├── Makefile                    # Build system (set CUDA_ARCH for your GPU)
├── submit_ncsu_hpc.sh          # LSF job script for NCSU HPC
├── results.md                  # Raw benchmark numbers from H100
└── report.md                   # This file
```

### Running

```bash
# On any machine with an NVIDIA GPU:
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader  # find your sm_XX
# Edit CUDA_ARCH in Makefile, then:
make clean && make all && make run

# NCSU HPC (LSF):
bqueues                   # find GPU queue name
bsub < submit_ncsu_hpc.sh
bjobs                     # monitor
```
