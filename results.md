# Benchmark Results — NVIDIA H100 SXM

**GPU:** NVIDIA H100 SXM  
**Architecture:** sm_90 (Hopper)  
**Memory:** 80 GB HBM3, ~3.35 TB/s bandwidth  
**Compiled:** `nvcc -O3 -std=c++17 -arch=sm_90`  
**Date:** April 2026

---

## Exponential Kernel (Compute-Bound)

1M elements, 512 exp() calls per thread in registers (register-resident, no global memory between calls).  
Timing: average of 20 runs after 3 warmup runs.  
RMSE: measured from a separate single-pass accuracy kernel vs CPU `expf()` reference.

| Method | Time (ms) | Speedup | RMSE |
|--------|-----------|---------|------|
| Hardware `__expf` | 0.1440 | 1.00x | 5.59e-08 |
| Polynomial Degree 3 | 0.0948 | **1.52x** | 2.17e-04 |
| Polynomial Degree 4 | 0.1115 | **1.29x** | 1.37e-05 |
| Polynomial Degree 5 | 0.1282 | **1.12x** | 7.34e-07 |

**Observation:** Higher degree = better accuracy but fewer FMAs means less advantage over hardware. Degree 4 is the best accuracy/speed tradeoff — matches Flash Attention 4's recommendation.

---

## Softmax Accuracy Test (Compute-Bound)

32 sequences × 10,000 vocab logits. Accuracy measured from a single-pass accuracy kernel. Timing from 32 internal softmax passes with data held in shared memory (warp-shuffle reduction, no atomicAdd).

**Accuracy (single pass, polynomial degree 4 vs hardware):**

| Metric | Value |
|--------|-------|
| Avg KL Divergence | ~0 (−9.31e-09, floating point noise) |
| Max KL Divergence | 8.43e-08 |
| Top-1 Match Rate | **100%** (32/32) |

**Compute-bound timing (32 internal iterations, shared memory):**

| Method | Time (ms) | Speedup |
|--------|-----------|---------|
| Hardware softmax | 0.1560 | 1.00x |
| Polynomial degree 4 softmax | 0.1357 | **1.15x** |

---

## Key Takeaway

The polynomial speedup is **architecture-specific**. The same benchmark on A100 showed no speedup (polynomial was slower or equal). On H100, the ratio of FMA throughput to hardware `ex2` instruction throughput favors the polynomial — which is exactly why Flash Attention 4 targets Hopper specifically.

| GPU | Exp kernel speedup (degree 4) | Softmax speedup |
|-----|-------------------------------|-----------------|
| A100 (sm_80) | ~0.91x (hardware wins) | ~0.93x |
| H100 (sm_90) | **1.29x** | **1.15x** |
