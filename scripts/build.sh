#! /bin/sh

# Go to the root of the project

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Ensure latest binaries are built
if [ ! -d "build" ]; then
  # create build directory and run cmake

  if [ "$(uname)" == "Darwin" ]; then
    echo "Running on macOS, using Homebrew LLVM"
    cmake -S . -B build -DCMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang"
  else
    echo "Running on Other Systems, using system default compiler"
    cmake -S . -B build
  fi

fi

cmake --build build

# Check if the model exists before quantizing
if [ ! -f "models/stories110M.pt" ]; then
  echo "ExitingL Model not found. Please download the model from https://huggingface.co/Stories/stories110M and place it in the models directory."
fi

# Check if fp32, int8 and fp16 models exist, if not, build them
if [ ! -f "models/stories110M.bin" ]; then
  echo "FP32 model not found. Creating it..."
  /opt/tinyllm-export/bin/python tools/export.py \
    models/stories110M.bin \
    --version 2 \
    --checkpoint models/stories110M.pt
fi

if [ ! -f "models/stories110M_int8.bin" ]; then
  echo "INT8 model not found. Creating it..."
  /opt/tinyllm-export/bin/python tools/export.py \
    models/stories110M_int8.bin \
    --version 2 \
    --checkpoint models/stories110M.pt
fi

if [ ! -f "models/stories110M_fp16.bin" ]; then
    echo "FP16 model not found. Creating it..."
    /opt/tinyllm-export/bin/python tools/export.py \
        models/stories110M_fp16.bin \
        --version 3 \
        --dtype fp16 \
        --checkpoint models/stories110M.pt || {
        echo "FP16 export failed"
        exit 1
    }
fi
