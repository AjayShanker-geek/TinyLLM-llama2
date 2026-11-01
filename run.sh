#! /bin/sh

# Ensure latest binaries are built
if [ ! -d "build" ]; then
  # create build directory and run cmake

  if [ "$(uname)" == "Darwin" ]; then
    echo "Running on macOS, using Homebrew LLVM"
    cmake -S . -B build -DCMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang"
  else
    echo "Running on Other Systems, using system default compiler"
    cmake -S . -B build
    cmake --build build --target run_int8_baseline
    cmake -S . -B build
    cmake --build build --target run_fp16_baseline

  fi

fi

cmake --build build

# Run the binaries
echo "Running the binaries..."

./build/run_fp32_baseline || {
  echo "run_fp32_baseline failed"
  exit 1
}
sleep 2

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

echo "Running run_fp16_baseline with input 'Once upon a time'..."
./build/run_fp16_baseline models/stories110M_fp16.bin \
    -z models/tokenizer.bin \
    -t 0 \
    -n 128 \
    -i "Once upon a time" || {
    echo "run_fp16_baseline failed"
    exit 1
}

if [ ! -f "models/stories110M_int8.bin" ]; then
    echo "INT8 model not found. Creating it..."
    /opt/tinyllm-export/bin/python tools/export.py \
        models/stories110M_int8.bin \
        --version 2 \
        --checkpoint models/stories110M.pt
fi
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
./build/run_fp32_openmp || {
  echo "run_fp32_openmp failed"
  exit 1
}

echo "All binaries executed successfully."
