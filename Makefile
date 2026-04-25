# Simple Makefile for CUDA exponential benchmark
# Usage: make all

NVCC = nvcc
CUDA_ARCH = -arch=sm_90  # Change based on your GPU: sm_70 (V100), sm_80 (A100), sm_90 (H100)
NVCC_FLAGS = -O3 -std=c++17 -lineinfo $(CUDA_ARCH)
NVCC_LIBS = -lcuda -lcudart

# Targets
BIN_DIR = ./bin
TARGETS = $(BIN_DIR)/exponential_kernel $(BIN_DIR)/softmax_accuracy_test

# Default target
all: $(TARGETS)

# Create bin directory
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Exponential kernel benchmark
$(BIN_DIR)/exponential_kernel: exponential_kernel.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)
	@echo "✓ Built exponential_kernel"

# Softmax accuracy test
$(BIN_DIR)/softmax_accuracy_test: softmax_accuracy_test.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(NVCC_LIBS)
	@echo "✓ Built softmax_accuracy_test"

# Run benchmarks
run: all
	@echo "================================"
	@echo "Running Exponential Kernel Bench"
	@echo "================================"
	$(BIN_DIR)/exponential_kernel
	@echo ""
	@echo "================================"
	@echo "Running Softmax Accuracy Test"
	@echo "================================"
	$(BIN_DIR)/softmax_accuracy_test
	@echo ""
	@echo "✓ All benchmarks completed"

# Plot results (requires Python + matplotlib + numpy)
plot: run
	python3 plot_results.py

# Clean build artifacts
clean:
	rm -rf $(BIN_DIR) *.csv *.png
	@echo "✓ Cleaned build artifacts"

# Help
help:
	@echo "CUDA Exponential Approximation Benchmark"
	@echo "========================================"
	@echo "Targets:"
	@echo "  make all    - Build all executables"
	@echo "  make run    - Build and run benchmarks"
	@echo "  make plot   - Run benchmarks and generate plots"
	@echo "  make clean  - Remove build artifacts"
	@echo "  make help   - Show this help message"
	@echo ""
	@echo "Configuration:"
	@echo "  GPU Compute Capability: $(CUDA_ARCH)"
	@echo "  Edit CUDA_ARCH if your GPU differs:"
	@echo "    - V100, A100: -arch=sm_80"
	@echo "    - A100: -arch=sm_80"
	@echo "    - RTX 20xx: -arch=sm_80"
	@echo "    - RTX 30xx: -arch=sm_80"
	@echo "    - L40: -arch=sm_80"

.PHONY: all run plot clean help
