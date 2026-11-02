# TinyLLM-llama2

Analyse and benchmark text generation for small pretrained Llama-2-style language models in C!

## Project Setup

Project structure reference from [Link](https://github.com/cpp-best-practices/cmake_template)

- Using CMake for build configuration for creating three binary for each sub-tasks
- `logs` directory is used to store the logs generated during the execution of the programs.
- `plots` uses python's matplotlib library to visualize the performance of the algorithms.
- `figures` directory is used to store the generated plots and figures.
- `build` directory is used to store the compiled binaries and other build artifacts.
- `src` directory contains the source code for the project, organized into subdirectories for each sub-task.
- `include` directory contains the header files for the project, organized into subdirectories for each sub-task.

## INT8 baseline (Linux / Docker)

The INT8 baseline runs the model in C using scalar, single-threaded matrix-vector multiplication. Weights and activation vectors use grouped INT8 quantization with FP32 scales. Products accumulate in INT32, and the output remains FP32.

Run the following commands from the project root inside the Docker container.

### Required files

- src/run_int8_baseline.c
- include/run_int8_baseline.h
- tools/llama2_export/export.py
- tools/llama2_export/model.py
- models/stories110M.pt
- models/tokenizer.bin (matching the model)

Make sure stories110M.pt is copied over correctly before doing the model.

### One-time Python setup for model conversion

```bash
# Ensure you have Python 3, pip and virtualenv installed
python3 -m venv venv
source venv/bin/activate
python3 -m pip install torch numpy
```

### Build and run

Manually, run in terminal

```bash
cmake -S . -B build
cmake --build build --target run_int8_baseline

./build-docker/run_int8_baseline models/stories110M_int8.bin \
    -z models/tokenizer.bin \
    -t 0 \
    -n 128 \
    -i "Once upon a time"
```

- -z selects the tokenizer.
- -t 0 uses greedy token selection.
- -n 128 allows up to 128 forward steps, including prompt processing.
- -i supplies the starting prompt.

Alternatively, run in it terminal

```bash

./scripts/build.sh
./scripts/run.sh
```

## Building the Project

```bash

# Check the Clang version & install CMake if not already installed
./scripts/build.sh
./scripts/run.sh
```

## Profile & Benchmark

Docker is used to ensure a consistent environment for profiling and benchmarking. The following commands will run the profiling and benchmarking scripts.

> [!IMPORTANT]
> Docker Desktop must be given more than these limits of CPU 10 and memory 8GB to run the profiling and benchmarking scripts.

```bash
# Build the Docker image for profiling and benchmarking
docker compose build

# Run the profiling script inside the Docker container
docker compose run --rm -e RUNS=10 bench bash scripts/profile.sh

```

The profiling script will generate logs in the `logs/profile` directory, which can be used to analyze the performance of the model.

### Benchmark System

- OS: Ubuntu 24.04 LTS
- CPU: 8 (aarch64 (Arm 64-bit))
- RAM: 4 GB

## Generate Plots

```bash
# Create a virtual environment for Python dependencies
python3 -m venv venv
source venv/bin/activate

pip3 install -r requirements.txt

# Run the plotting script to generate performance plots
python3 plots/plot.py
```

## Acknowledgments

- This project is inspired by and reference on [karpathy/llama2.c](https://github.com/karpathy/llama2.c)
