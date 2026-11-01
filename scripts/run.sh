#! /bin/sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Run the binaries
echo "Running the binaries..."

./build/run_fp32_baseline || {
  echo "run_fp32_baseline failed"
  exit 1
}

sleep 2

echo "Running run_int8_baseline with input 'Once upon a time'..."s
./build/run_int8_baseline models/stories110M_int8.bin \
  -z models/tokenizer.bin \
  -t 0 \
  -n 128 \
  -i "Once upon a time" || {
  echo "run_int8_baseline failed"
  exit 1
}

sleep 2

./build/run_fp32_simd || {
  echo "run_fp32_simd failed"
  exit 1
}

sleep 2

./build/run_fp32_openmp ./models/stories110M.bin -i "Once upon a time" -z ./models/tokenizer.bin -t 0 || {
  echo "run_fp32_openmp failed"
  exit 1
}

echo "All binaries executed successfully."
