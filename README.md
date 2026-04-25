# CUDA Exponential Approximation Benchmark

Benchmarks polynomial approximations of `exp()` on CUDA vs. hardware `__expf`, measuring speedup and accuracy. Inspired by Flash Attention 4's optimization technique.

## Setup

### 1. Find your GPU's compute capability

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

Edit `CUDA_ARCH` in [Makefile](Makefile) to match:

| GPU | `CUDA_ARCH` |
|-----|------------|
| V100 | `sm_70` |
| A100 | `sm_80` |
| RTX 20xx | `sm_75` |
| RTX 30xx / A5000 | `sm_86` |
| L40 / RTX 40xx | `sm_89` |

### 2. Build and run

```bash
make all        # compile
make run        # run both benchmarks
python3 plot_results.py  # generate plots (requires matplotlib, numpy)
```

---

## NCSU HPC (LSF)

```bash
ssh <unity_id>@login.hpc.ncsu.edu
cd /path/to/cuda_exponential_benchmark

# Check available GPU queues
bqueues

# Submit job
bsub < submit_ncsu_hpc.sh

# Monitor status
bjobs

# View output when done (LSF writes to files specified in the script)
cat job_output.log
```

> The submit script (`submit_ncsu_hpc.sh`) handles loading CUDA modules, building, and running automatically.

---

## Direct SSH into a GPU machine

```bash
ssh user@hostname

# Verify GPU
nvidia-smi

# Load CUDA if needed (module-managed systems)
module load cuda   # or: module avail cuda to find the right version

# Build and run
cd /path/to/cuda_exponential_benchmark
make run
```

---

## Output

After running, you'll have:
- `benchmark_results.csv` — timing and RMSE per method
- `softmax_accuracy_results.csv` — KL divergence and top-1 match rates
- `accuracy_comparison.png` / `detailed_analysis.png` — plots (if Python available)

### What to expect

| Method | Speedup | RMSE |
|--------|---------|------|
| `__expf` (baseline) | 1.00x | 0 |
| Poly degree 3 | ~1.3x | ~3.5e-3 |
| Poly degree 4 | ~1.5x | ~2.1e-4 |
| Poly degree 5 | ~1.6x | ~1.2e-5 |

---

## Troubleshooting

**`cannot find -lcuda` or `nvcc: not found`**
```bash
module load cuda   # on module-managed systems
```

**`Unsupported gpu architecture`**
```bash
nvidia-smi  # check GPU, then update CUDA_ARCH in Makefile
make clean && make all
```

**Out of memory** — reduce array size in `exponential_kernel.cu` near `int n = 1024 * 1024`
