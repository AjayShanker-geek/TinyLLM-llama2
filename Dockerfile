# Benchmark/profiling environment. Models are mounted from ./models, not exported here.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build gdb binutils \
        valgrind linux-tools-generic linux-tools-common \
        hyperfine time util-linux \
        # cg_annotate (cachegrind's report tool) is a Python 3 script
        python3 \
    && rm -rf /var/lib/apt/lists/* \
    # Docker Desktop's kernel never matches Ubuntu's perf wrapper; use the binary directly
    && ln -sf "$(ls /usr/lib/linux-tools/*/perf | head -n 1)" /usr/local/bin/perf

WORKDIR /workspace
CMD ["bash"]
