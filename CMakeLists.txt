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

# -march for the FP16 builds. On ARM64, +fp16 = FP16 arithmetic (scalar and NEON) and
# +fp16fml = FMLAL/FMLSL (FP16 multiply, FP32 accumulate); every Apple M-series, Graviton 2+
# and Cortex-A76+ has both. A CPU without them crashes with "Illegal instruction".
# On x86-64, x86-64-v3 includes F16C (hardware FP16 <-> FP32 conversion).
# Check the CPU in Docker: grep -o -w 'fphp\|asimdhp\|asimdfhm' /proc/cpuinfo | sort -u
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(DEFAULT_FP16_ARCH "x86-64-v3")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  set(DEFAULT_FP16_ARCH "armv8.2-a+fp16fml")
else()
  set(DEFAULT_FP16_ARCH "native")
endif()
set(FP16_ARCH "${DEFAULT_FP16_ARCH}" CACHE STRING "Value passed to -march for the FP16 builds (scalar, no SIMD)")

include_directories(include)

# Ensure no optimzation done by the compiler
add_library(no_optimization INTERFACE)
add_library(fp16_optimisation INTERFACE)
add_library(simd_optimisation INTERFACE)
add_library(openmp_optimisation INTERFACE)
add_library(openmp_simd_aggressive_optimisation INTERFACE)

target_compile_options(no_optimization
    INTERFACE
        # $<$<CONFIG:Debug>:-O0>
        # $<$<CONFIG:Debug>:-g>
        -O2 # Disable compiler optimizations
        -fno-tree-vectorize # Disable vectorization
        -Wall
        -Wextra
        -Wpedantic
)

# Same as no_optimization, plus -march with the FP16 extensions (FP16_ARCH).
# Scalar only: the FP16 builds must not use SIMD, so both auto-vectorizers are off
# (loop vectorizer and SLP, which packs neighbouring scalar operations into vectors).
# -march only makes the FP16 instructions available; the code stays scalar.
target_compile_options(fp16_optimisation
    INTERFACE
        -O2 # same as the baseline
        -march=${FP16_ARCH} # FP16 arithmetic + FMLAL on ARM64, F16C on x86-64
        -fno-tree-vectorize # No loop auto-vectorization, same as the baseline
        -fno-tree-slp-vectorize # No SLP (straight-line) vectorization either
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(simd_optimisation
    INTERFACE
        -O2 # Disable compiler optimizations, same as the baseline
        -march=${SIMD_ARCH} # Enable architecture-specific optimizations
        -fno-tree-vectorize # No auto-vectorization: only the hand-written NEON is SIMD
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(openmp_optimisation
    INTERFACE
        -O2 # Enable compiler optimizations
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
    include/tokenizer.h
    include/perplexity.h
    src/run_fp32_baseline.c
    src/tokenizer.c
    src/perplexity.c
)

target_include_directories(run_fp32_baseline PRIVATE include)

target_link_libraries(run_fp32_baseline PRIVATE no_optimization)

# Build for FP16 binary
add_executable(run_fp16_baseline
    include/run_fp16_baseline.h
    include/tokenizer.h
    include/perplexity.h
    src/run_fp16_baseline.c
    src/tokenizer.c
    src/perplexity.c
)

target_include_directories(run_fp16_baseline PRIVATE include)

target_link_libraries(run_fp16_baseline PRIVATE fp16_optimisation)

# Build for FP16 binary using the _Float16 type (hardware FP16->FP32 conversion on ARM64)
add_executable(run_fp16_baseline_native
    include/run_fp16_baseline_native.h
    include/tokenizer.h
    include/perplexity.h
    src/run_fp16_baseline_native.c
    src/tokenizer.c
    src/perplexity.c
)

target_link_libraries(run_fp16_baseline_native PRIVATE fp16_optimisation)


# Build for INT8 binary
add_executable(run_int8_baseline
    include/run_int8_baseline.h
    include/tokenizer.h
    include/perplexity.h
    src/run_int8_baseline.c
    src/tokenizer.c
    src/perplexity.c
)

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

# Build for OpenMP + NEON SIMD binary (4-row NEON kernel inside OpenMP threads)
add_executable(run_fp32_openmp_neon
    include/run_fp32_openmp_neon.h
    include/tokenizer.h
    include/util.h
    include/sampler.h
    include/perplexity.h
    src/run_fp32_openmp_neon.c
    src/tokenizer.c
    src/util.c
    src/perplexity.c
)

target_link_libraries(run_fp32_openmp_neon
    PRIVATE
        openmp_optimisation
)

foreach(lib no_optimization fp16_optimisation simd_optimisation openmp_optimisation openmp_simd_aggressive_optimisation)
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

# BEGIN NEXT TOKEN AGREEMENT
# Paste at the END of the root CMakeLists.txt, after runner targets are defined.
foreach(agreement_target IN ITEMS
    run_fp32_baseline run_int8_baseline run_fp16_baseline
    run_fp16_baseline_native run_fp32_simd run_fp32_simd_4row
    run_fp32_openmp run_fp32_openmp_neon run_fp32_openmp_simd)
  if(TARGET ${agreement_target})
    get_target_property(agreement_added ${agreement_target} NEXT_TOKEN_AGREEMENT_ADDED)
    if(NOT agreement_added)
      target_sources(${agreement_target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/agreement.c")
      target_include_directories(${agreement_target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
      set_property(TARGET ${agreement_target} PROPERTY NEXT_TOKEN_AGREEMENT_ADDED TRUE)
    endif()
  endif()
endforeach()

if(NOT TARGET compare_next_tokens)
  add_executable(compare_next_tokens src/compare_next_tokens.c)
  target_compile_features(compare_next_tokens PRIVATE c_std_11)
endif()
# END NEXT TOKEN AGREEMENT
