#!/bin/bash
#BSUB -J cuda_exp_bench
#BSUB -W 00:15
#BSUB -n 4
#BSUB -R "rusage[mem=16GB] select[gpu]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpu_access          # run: bqueues  to find your GPU queue name
#BSUB -o job_output.log
#BSUB -e job_error.log

# Usage: bsub < submit_ncsu_hpc.sh

echo "Job: $LSB_JOBID  |  Node: $HOSTNAME  |  GPU: $CUDA_VISIBLE_DEVICES"
echo "Started: $(date)"
echo ""

# Load CUDA — check available versions with: module avail cuda
module load cuda/12.2

nvcc --version
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Build
cd "$LS_SUBCWD" || exit 1
make clean && make all || { echo "ERROR: Build failed"; exit 1; }

# Run
echo "--- Exponential Kernel Benchmark ---"
./bin/exponential_kernel

echo ""
echo "--- Softmax Accuracy Test ---"
./bin/softmax_accuracy_test

# Plots (optional)
if command -v python3 &>/dev/null; then
    python3 plot_results.py && echo "Plots generated."
fi

echo ""
echo "Output files:"
ls -lh *.csv *.png 2>/dev/null

echo ""
echo "Done: $(date)"
