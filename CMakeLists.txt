cmake_minimum_required(VERSION 4.4.3)

project(llama2_C
  VERSION 0.0.1
  LANGUAGES C
)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

# Ensure no optimzation done by the compiler
add_library(no_optimization INTERFACE)
add_library(simd_optimisaton INTERFACE)
add_library(openmp_optimisation INTERFACE)
add_library(openmp_simd_aggressive_optimisation INTERFACE)

target_compile_options(no_optimization
    INTERFACE
        # $<$<CONFIG:Debug>:-O0>
        # $<$<CONFIG:Debug>:-g>
        -O0 # Disable compiler optimizations
        -fno-tree-vectorize # Disable vectorization
        -pg # Enable profiling with gprof
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(simd_optimisaton
    INTERFACE
        -O0 # Enable compiler optimizations
        -pg # Enable profiling with gprof
        -march=native # Enable architecture-specific optimizations
        -ftree-vectorize # Enable vectorization
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(openmp_optimisation
    INTERFACE
        -O0 # Enable compiler optimizations
        -pg # Enable profiling with gprof
        -fno-tree-vectorize # Disable vectorization
        -fopenmp # Enable OpenMP support
        -march=native # Enable architecture-specific optimizations
        -Wall
        -Wextra
        -Wpedantic
)

# Ensure aggressive optimizations are done by the compiler with OpenMP and SIMD
target_compile_options(openmp_simd_aggressive_optimisation
    INTERFACE
        -O3 # Enable compiler optimizations
        -pg # Enable profiling with gprof
        -ftree-vectorize # Enable vectorization
        -fopenmp # Enable OpenMP support
        -march=native # Enable architecture-specific optimizations
        -Wall
        -Wextra
        -Wpedantic
)

# Build for FP32 baseline binary
add_executable(run_fp32_baseline
    include/run_fp32_baseline.h
    src/run_fp32_baseline.c
)

target_link_libraries(run_fp32_baseline
    PRIVATE
        no_optimization
)

# Build for FP16 binary
add_executable(run_fp16_baseline
    src/run_fp16_baseline.c
    include/run_fp16_baseline.h
)

target_include_directories(run_fp16_baseline PRIVATE include)

target_link_libraries(run_fp16_baseline PRIVATE no_optimization)

if(NOT WIN32)
    target_link_libraries(run_fp16_baseline PRIVATE m)
endif()


# Build for INT8 binary
add_executable(run_int8_baseline
    include/run_int8_baseline.h
    src/run_int8_baseline.c
)

if(NOT WIN32)
  target_link_libraries(run_int8_baseline PRIVATE m)
endif()

target_link_libraries(run_int8_baseline
    PRIVATE
        no_optimization
)

# Build for SIMD binary
add_executable(run_fp32_simd
    include/run_fp32_simd.h
    src/run_fp32_simd.c
)

target_link_libraries(run_fp32_simd
    PRIVATE
        simd_optimisaton
)

# Build for OpenMP binary
add_executable(run_fp32_openmp
    include/run_fp32_openmp.h
    include/tokenizer.h
    include/util.h
    include/sampler.h
    src/run_fp32_openmp.c
    src/tokenizer.c
    src/util.c
)

# -fopenmp must also be passed at link time to pull in the OpenMP runtime (libomp)
target_link_options(openmp_optimisation INTERFACE -fopenmp)
target_link_libraries(run_fp32_openmp
    PRIVATE
        openmp_optimisation
)

add_executable(run_fp32_openmp_simd
    include/run_fp32_openmp.h
    include/tokenizer.h
    include/util.h
    include/sampler.h
    src/run_fp32_openmp.c
    src/tokenizer.c
    src/util.c
)

target_link_options(openmp_simd_aggressive_optimisation INTERFACE -fopenmp)
target_link_libraries(run_fp32_openmp_simd
    PRIVATE
        openmp_simd_aggressive_optimisation
)
