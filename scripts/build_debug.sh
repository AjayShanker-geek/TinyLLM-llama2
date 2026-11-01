#! /bin/sh

set -euo pipefail

# Go to the root of the project
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Ensure latest binaries are built
if [ ! -d "build/debug" ]; then
  # create build directory and run cmake

  if [ "$(uname)" == "Darwin" ]; then
    echo "Running on macOS, using Homebrew LLVM"
    cmake -S . -B build/debug -DCMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" -DCMAKE_BUILD_TYPE=Debug -DENABLE_GPROF=ON
  else
    echo "Running on Other Systems, using system default compiler"
    cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Debug -DENABLE_GPROF=ON
  fi

fi

cmake --build build/debug 
