cmake_minimum_required(VERSION 3.18)
project(cuda_exponential_benchmark LANGUAGES CUDA CXX)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Compilation flags for better optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")

# Add executable for exponential kernel benchmark
add_executable(exponential_kernel exponential_kernel.cu)
target_link_libraries(exponential_kernel PRIVATE cuda_runtime)

# Add executable for softmax accuracy test
add_executable(softmax_accuracy_test softmax_accuracy_test.cu)
target_link_libraries(softmax_accuracy_test PRIVATE cuda_runtime)

# Set output directory
set_target_properties(exponential_kernel PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set_target_properties(softmax_accuracy_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Print configuration
message(STATUS "CUDA Toolkit Path: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA Compute Capability: ${CUDA_COMPUTE_CAPABILITY}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
