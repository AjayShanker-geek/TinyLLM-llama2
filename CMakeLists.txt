cmake_minimum_required(VERSION 3.25)

project(llama2_C
  VERSION 0.0.1
  LANGUAGES C
)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# gprof instr adds overhead, so keep it out of timing/perf/valgrind builds
option(ENABLE_GPROF "Instrument binaries with -pg for gprof" OFF)

# Fixed target ISA instead of -march=native, so every machine of the same
# architecture builds identical SIMD code
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(DEFAULT_SIMD_ARCH "x86-64-v3") # AVX2 + FMA
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  set(DEFAULT_SIMD_ARCH "armv8.2-a") # NEON
else()
  set(DEFAULT_SIMD_ARCH "native")
endif()
set(SIMD_ARCH "${DEFAULT_SIMD_ARCH}" CACHE STRING "Value passed to -march for SIMD/OpenMP builds")

include_directories(include)

# Ensure no optimzation done by the compiler
add_library(no_optimization INTERFACE)
add_library(simd_optimisation INTERFACE)
add_library(openmp_optimisation INTERFACE)
add_library(openmp_simd_aggressive_optimisation INTERFACE)

target_compile_options(no_optimization
    INTERFACE
        # $<$<CONFIG:Debug>:-O0>
        # $<$<CONFIG:Debug>:-g>
        -O0 # Disable compiler optimizations
        -fno-tree-vectorize # Disable vectorization
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(simd_optimisation
    INTERFACE
        -O0 # Disable compiler optimizations, same as the baseline
        -march=${SIMD_ARCH} # Enable architecture-specific optimizations
        -fno-tree-vectorize # No auto-vectorization: only the hand-written NEON is SIMD
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(openmp_optimisation
    INTERFACE
        -O0 # Enable compiler optimizations
        -fno-tree-vectorize # Disable vectorization
        -fopenmp # Enable OpenMP support
        -march=${SIMD_ARCH} # Enable architecture-specific optimizations
        -Wall
        -Wextra
        -Wpedantic
)

# Ensure aggressive optimizations are done by the compiler with OpenMP and SIMD
target_compile_options(openmp_simd_aggressive_optimisation
    INTERFACE
        -O3 # Enable compiler optimizations
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
    include/tokenizer.h
    include/perplexity.h
    src/run_int8_baseline.c
    src/tokenizer.c
    src/perplexity.c
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
    include/tokenizer.h
    include/perplexity.h
    src/run_fp32_simd.c
    src/tokenizer.c
    src/perplexity.c
)

target_link_libraries(run_fp32_simd
    PRIVATE
        simd_optimisation
)

# Build for SIMD binary with the 4-row kernel (same source, -DSIMD_4ROW)
add_executable(run_fp32_simd_4row
    include/run_fp32_simd.h
    include/tokenizer.h
    include/perplexity.h
    src/run_fp32_simd.c
    src/tokenizer.c
    src/perplexity.c
)

target_compile_definitions(run_fp32_simd_4row PRIVATE SIMD_4ROW)

target_link_libraries(run_fp32_simd_4row
    PRIVATE
        simd_optimisation
)

# Kernel correctness test for the SIMD matmul kernels
add_executable(test_matmul_simd
    tests/test_matmul_simd.c
    src/tokenizer.c
)

target_link_libraries(test_matmul_simd
    PRIVATE
        simd_optimisation
)

# Build for OpenMP binary
add_executable(run_fp32_openmp
    include/run_fp32_openmp.h
    include/tokenizer.h
    include/util.h
    include/sampler.h
    include/perplexity.h
    src/run_fp32_openmp.c
    src/tokenizer.c
    src/util.c
    src/perplexity.c
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
    include/perplexity.h
    src/run_fp32_openmp.c
    src/tokenizer.c
    src/util.c
    src/perplexity.c
)

target_link_options(openmp_simd_aggressive_optimisation INTERFACE -fopenmp)
target_link_libraries(run_fp32_openmp_simd
    PRIVATE
        openmp_simd_aggressive_optimisation
)

foreach(lib no_optimization simd_optimisation openmp_optimisation openmp_simd_aggressive_optimisation)
  # Debug symbols so perf/valgrind can show function names and lines
  target_compile_options(${lib} INTERFACE -g)
  if(NOT WIN32)
    # sqrtf/expf/powf live in libm on Linux (macOS has them in libSystem)
    target_link_libraries(${lib} INTERFACE m)
  endif()
  if(ENABLE_GPROF)
    # -pg is needed at link time too, otherwise gmon.out is never written
    target_compile_options(${lib} INTERFACE -pg)
    target_link_options(${lib} INTERFACE -pg)
  endif()
endforeach()
