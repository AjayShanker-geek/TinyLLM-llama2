# TinyLLM-llama2

Analyse and benchmark text generation for small pretrained Llama-2-style language models in C!

## Project Setup

The project structure follows the [cpp-best-practices CMake template](https://github.com/cpp-best-practices/cmake_template).

- `CMakeLists.txt` builds nine inference binaries from shared code: the FP32, FP16 (software decode and `_Float16`) and INT8 baselines, FP32 SIMD (1-row and 4-row NEON), and FP32 OpenMP (scalar, `-O3` auto-vectorised and NEON). It also builds the `test_matmul_simd` unit test and the `compare_next_tokens` tool.
- `src/` holds the C source: one `run_*.c` file per binary, plus the shared `tokenizer.c`, `util.c`, `perplexity.c` and `agreement.c`.
- `include/` holds the matching headers, one per source file, plus `sampler.h` and `transformer_fp32.h`.
- `tests/` holds `test_matmul_simd.c`, which checks the NEON matmul kernels against the scalar version.
- `tools/` holds `export.py` and `model.py`, which convert a PyTorch checkpoint (`.pt`) into the FP32, FP16 and INT8 model files.
- `models/` holds the model checkpoints, exported `.bin` files and `tokenizer.bin` (not tracked by git).
- `data/` holds the benchmark prompts (`prompts.txt`), the perplexity text (`eval.txt`) and the TinyStories validation set (`TinyStories-valid.txt`), which supplies the prompts and the next-token agreement stories.
- `scripts/` holds the build and run scripts, the profiling pipeline (`profile.sh`), the next-token agreement script (`agreement.sh`) and the prompt generator (`make_prompts.py`).
- `Dockerfile` and `compose.yaml` define the benchmark container: Ubuntu 24.04 aarch64, 8 CPUs, 4 GB RAM, with perf and valgrind installed.
- `build/` holds the local build. `build-docker/` and `build-docker-gprof/` hold the container builds made by `profile.sh` (normal and `-pg`). None are tracked by git.
- `logs/profile/<run-id>/` holds each profiling run: `results.csv`, `perplexity.csv`, `next_token_agreement.csv`, `system.txt`, one folder of profiler output per binary, and the generated `figures/` and `tables/`.
- `plots/` holds the Python analysis scripts: `data_summary.py` loads a run, `plots.py` draws the figures and `main_table.py` builds the summary table.
- `report/` holds the engineering report and the presentation slides.

## INT8 baseline (Linux / Docker)

The INT8 baseline runs the model in C using scalar, single-threaded matrix-vector multiplication. Weights and activation vectors use grouped INT8 quantization with FP32 scales. Products accumulate in INT32, and the output remains FP32.

Run the following commands from the project root inside the Docker container.

### Required files

- src/run_int8_baseline.c
- include/run_int8_baseline.h
- tools/export.py
- tools/model.py
- models/stories110M.pt
- models/tokenizer.bin (matching the model)

Make sure `models/stories110M.pt` has been copied over completely before exporting the model.

### One-time Python setup for model conversion

```bash
# Ensure you have Python 3 and pip installed
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install torch numpy
```

### Export the model files

`tools/export.py` converts the PyTorch checkpoint into the `.bin` formats the C binaries read:

```bash
# FP32 (version 0): used by every FP32 binary
python3 tools/export.py models/stories110M.bin --version 0 --checkpoint models/stories110M.pt

# INT8 (version 2): grouped INT8 weights with FP32 scales
python3 tools/export.py models/stories110M_int8.bin --version 2 --checkpoint models/stories110M.pt

# FP16 (version 3): used by both FP16 binaries
python3 tools/export.py models/stories110M_fp16.bin --version 3 --checkpoint models/stories110M.pt
```

### Build and run

To build and run manually, use these commands:

```bash
cmake -S . -B build
cmake --build build --target run_int8_baseline

./build/run_int8_baseline models/stories110M_int8.bin \
    -z models/tokenizer.bin \
    -t 0 \
    -n 128 \
    -i "Once upon a time"
```

- -z selects the tokenizer.
- -t 0 uses greedy token selection.
- -n 128 allows up to 128 forward steps, including prompt processing.
- -i supplies the starting prompt.

Alternatively, use the helper scripts:

```bash
./scripts/build.sh
./scripts/run.sh
```

## Building the Project

`scripts/build.sh` configures CMake into `build/` (on macOS it uses Homebrew LLVM Clang) and builds every target. `scripts/run.sh` runs the FP32 baseline and the FP16 `_Float16` baseline on a sample prompt. The other binaries are listed in the script, commented out.

```bash
# Requires CMake and a C compiler (Homebrew LLVM on macOS)
./scripts/build.sh
./scripts/run.sh
```

## Profile & Benchmark

Docker is used to ensure a consistent environment for profiling and benchmarking. The following commands will run the profiling and benchmarking scripts.

> [!IMPORTANT]
> Docker Desktop must be given at least 8 CPUs and 4 GB of memory. `compose.yaml` limits the container to 8 CPUs (`cpuset: "0-7"`) and 4 GB with no swap.

```bash
# Build the Docker image for profiling and benchmarking
docker compose build

# Run the profiling script inside the Docker container (RUNS = measured repeats per prompt, default 5)
docker compose run --rm -e RUNS=10 bench bash scripts/profile.sh

# Run the next-token agreement check on its own
docker compose run --rm bench bash scripts/agreement.sh
```

The profiling script writes each run to `logs/profile/<run-id>/`, which can be used to analyse the performance of the model.

### Benchmark System

- OS: Ubuntu 24.04 LTS
- CPU: 8 (aarch64 (Arm 64-bit))
- RAM: 4 GB

## Analysis Scripts

```bash
# Create a virtual environment for the analysis dependencies (matplotlib, pandas, numpy)
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

Every script below reads the latest run in `logs/profile/` by default. Pass a run folder to use a specific one.

```bash
# Print a summary of a run; --save also writes every table to <run-id>/tables/*.csv
.venv/bin/python plots/data_summary.py logs/profile/20261107-164852 --save

# Draw every figure into <run-id>/figures/ (PNG and PDF); --show also opens them
.venv/bin/python plots/plots.py logs/profile/20261107-164852

# Build the main results table (tok/s, speedup, size, bandwidth, perplexity, agreement)
# printed to stdout and written to <run-id>/tables/main_results.csv
.venv/bin/python plots/main_table.py logs/profile/20261107-164852
```

To regenerate the benchmark prompts from the TinyStories validation set (fixed seed, so the output is reproducible):

```bash
# Defaults: 20 prompts, seed 42, written to data/prompts.txt
python3 scripts/make_prompts.py -n 20 -s 42 -o data/prompts.txt
```

## Acknowledgments

- This project is inspired by and based on [karpathy/llama2.c](https://github.com/karpathy/llama2.c).
