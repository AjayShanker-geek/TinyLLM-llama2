#! /bin/sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Run the binaries
echo "Running the binaries..."

echo "Running run_fp32_baseline with input 'Once upon a time'..."
./build/run_fp32_baseline models/stories110M.bin \
  -z models/tokenizer.bin \
  -t 0 \
  -n 128 \
  -i "Once upon a time" || {
  echo "run_fp32_baseline failed"
  exit 1
}

sleep 2

echo "Running run_int8_baseline with input 'Once upon a time'..."
./build/run_int8_baseline models/stories110M_int8.bin \
  -z models/tokenizer.bin \
  -t 0 \
  -n 128 \
  -i "Once upon a time" || {
  echo "run_int8_baseline failed"
  exit 1
}

sleep 2

echo "Running run_fp16_baseline with input 'Once upon a time'..."
./build/run_fp16_baseline models/stories110M_fp16.bin \
  -z models/tokenizer.bin \
  -t 0 \
  -n 128 \
  -i "Once upon a time" || {
  echo "run_fp16_baseline failed"
  exit 1
}

sleep 2

echo "Running run_fp32_simd with input 'Once upon a time'..."
./build/run_fp32_simd models/stories110M.bin \
  -z models/tokenizer.bin \
  -t 0 \
  -n 128 \
  -i "Once upon a time" || {
  echo "run_fp32_simd failed"
  exit 1
}

sleep 2

echo "Running run_fp32_openmp with input 'Once upon a time'..."
./build/run_fp32_openmp models/stories110M.bin \
  -z models/tokenizer.bin \
  -t 0 \
  -n 128 \
  -i "Once upon a time" || {
  echo "run_fp32_openmp failed"
  exit 1
}

echo "All binaries executed successfully."
