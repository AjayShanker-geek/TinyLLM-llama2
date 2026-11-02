#!/usr/bin/env bash
# profile.sh - build and profile the benchmark binaries inside Docker
#
# Usage (from the project root, on any OS):
#   docker compose run --rm bench bash scripts/profile.sh                    # binaries in ALL_BINARIES
#   docker compose run --rm bench bash scripts/profile.sh run_int8_baseline  # only the ones named
#
# Settings are environment variables (see SETTINGS below), e.g.
#   docker compose run --rm -e RUNS=10 -e LABEL=nowait bench bash scripts/profile.sh
#
# What it does, for each binary:
#   1. timing        1 warm-up + RUNS runs: wall time and the binary's own tok/s
#   2. time -v       peak memory (max RSS), CPU %, page faults, context switches
#   3. perf stat     task clock, context switches, hardware counters if the machine has them
#   4. perf record   % of time spent in each function (all threads)
#   5. tool-specific steps (see the `case` at the bottom):
#        gprof       % of time per function + call counts (single-threaded builds only)
#        callgrind   exact instruction count per function
#        cachegrind  simulated cache misses (fixed cache size, same on every machine)
#        massif      peak heap memory
#        thread sweep  timing repeated for each thread count in THREADS_SWEEP (OpenMP)
#
# Output, in logs/profile/<run_id>[-<label>]/:
#   results.csv     every measurement of this run, one row each
#   system.txt      CPU, limits, compiler and settings of this run
#   <binary>/       raw tool output: runs/run_01.txt, gprof.txt, perf.data, ...
# Every run's rows are also appended to logs/profile/all_results.csv.
#
# CSV columns:
#   run_id   timestamp of this profile run      label    LABEL setting (free text)
#   arch     aarch64 / x86_64                   binary   e.g. run_fp32_openmp
#   tool     timing, time, perf_stat, ...       metric   e.g. wall_s, max_rss_kb, self_pct
#   item     function name (empty if n/a)       threads  OMP_NUM_THREADS used (1 if not OpenMP)
#   repeat   1..RUNS for timing rows            value    the number
#   unit     s, %, kB, count, ...
#
# Note: the binaries print tok/s differently (run_fp32_openmp includes sampling and
# printing, run_int8_baseline times only forward()), so compare binaries by wall_s.
#
# This script expects the Ubuntu container's GNU tools (date +%N, stat -c).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# SETTINGS (override with -e NAME=value)
LABEL="${LABEL:-}"                     # tag for this run, e.g. the code version measured
RUNS="${RUNS:-5}"                      # measured repeats per timing (after 1 warm-up)
STEPS="${STEPS:-128}"                  # tokens per run (-n)
VALGRIND_STEPS="${VALGRIND_STEPS:-16}" # tokens for valgrind runs (~50x slower)
THREADS_SWEEP="${THREADS_SWEEP:-1,2,4,8}"
PROMPT="${PROMPT:-Once upon a time}"
SEED="${SEED:-42}"
TOP_N="${TOP_N:-15}" # functions kept per profiler in the CSV

FP32_MODEL="${FP32_MODEL:-models/stories110M.bin}"
INT8_MODEL="${INT8_MODEL:-models/stories110M_int8.bin}"
TOKENIZER="${TOKENIZER:-models/tokenizer.bin}"

# ALL_BINARIES=(run_fp32_baseline run_int8_baseline run_fp32_simd run_fp32_openmp)
ALL_BINARIES=(run_int8_baseline run_fp32_simd run_fp32_simd_4row run_fp32_openmp run_fp32_openmp_simd)

# Fixed simulated cache for cachegrind: 32 KiB L1 I/D, 8 MiB last level
CACHEGRIND_CACHE=(--I1=32768,8,64 --D1=32768,8,64 --LL=8388608,16,64)

# PATHS
RUN_ID="$(date +%Y%m%d-%H%M%S)"
ARCH="$(uname -m)"

BUILD_DIR="$ROOT/build-docker"             # normal build: timing, perf, valgrind
GPROF_BUILD_DIR="$ROOT/build-docker-gprof" # -pg build: gprof only
LOG_ROOT="$ROOT/logs/profile/${RUN_ID}${LABEL:+-$LABEL}"
RESULTS_CSV="$LOG_ROOT/results.csv"
ALL_RESULTS_CSV="$ROOT/logs/profile/all_results.csv"
CSV_HEADER="run_id,label,arch,binary,tool,metric,item,threads,repeat,value,unit"

if (($#)); then BINARIES=("$@"); else BINARIES=("${ALL_BINARIES[@]}"); fi
FAILURES=()

##################################################################################
# HELPERS
##################################################################################
log() { printf '\n==> %s\n' "$*"; }

# step <name> <function>: run one profiling step. A failure is recorded and
# reported at the end, but does not stop the other steps.
# (set -e is off inside `if`, so the step functions chain commands with &&.)
step() {
  local name="$1"
  shift
  log "$name"
  if ! "$@"; then
    echo "!! $name failed" >&2
    FAILURES+=("$name")
  fi
}

model_for() {
  case "$1" in
  run_int8_baseline) echo "$INT8_MODEL" ;;
  *) echo "$FP32_MODEL" ;;
  esac
}

# Thread count a binary uses by default: OMP_NUM_THREADS for OpenMP builds, else 1
threads_for() {
  case "$1" in
  *openmp*) echo "${OMP_NUM_THREADS:-$(nproc)}" ;;
  *) echo 1 ;;
  esac
}

# Command-line arguments for a binary, one per line (read with mapfile)
args_for() {
  local bin="$1" steps="$2"
  printf '%s\n' "$ROOT/$(model_for "$bin")" -z "$ROOT/$TOKENIZER" \
    -t 0 -s "$SEED" -n "$steps" -i "$PROMPT"
}

# Check that the model and tokenizer files exist for each binary. If any are missing, print an error and return
check_inputs() {
  local missing=0 bin file
  for bin in "${BINARIES[@]}"; do
    for file in "$(model_for "$bin")" "$TOKENIZER"; do
      if [[ ! -f "$file" ]]; then
        echo "Missing $file (needed by $bin). Copy the shared model files into models/." >&2
        missing=1
      fi
    done
  done
  return "$missing"
}

build() {
  # Build binaries normal
  log "Building (normal) into ${BUILD_DIR#"$ROOT"/}"
  cmake -S . -B "$BUILD_DIR" -G Ninja -DENABLE_GPROF=OFF
  cmake --build "$BUILD_DIR" --target "${BINARIES[@]}"

  # Build binaries with -pg for gprof
  log "Building (gprof, -pg) into ${GPROF_BUILD_DIR#"$ROOT"/}"
  cmake -S . -B "$GPROF_BUILD_DIR" -G Ninja -DENABLE_GPROF=ON
  cmake --build "$GPROF_BUILD_DIR" --target "${BINARIES[@]}"
}

system_info() {
  {
    echo "# date"
    date -u
    echo "# uname"
    uname -a
    echo "# lscpu"
    lscpu
    echo "# nproc"
    nproc
    echo "# memory"
    free -m
    echo "# cgroup cpu.max"
    cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
    echo "# cgroup cpuset"
    cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || true
    echo "# cgroup memory.max"
    cat /sys/fs/cgroup/memory.max 2>/dev/null || true
    echo "# compiler"
    gcc --version | head -n 1
    echo "# cmake"
    cmake --version | head -n 1
    echo "# OpenMP env"
    env | grep -E '^OMP_' || true
    echo "# settings"
    echo "LABEL=$LABEL STEPS=$STEPS VALGRIND_STEPS=$VALGRIND_STEPS RUNS=$RUNS SEED=$SEED PROMPT=$PROMPT"
  } >"$LOG_ROOT/system.txt"
}

##################################################################################
# CSV
##################################################################################

csv_quote() { # wrap in quotes, doubling any quotes inside (standard CSV escaping)
  local s="${1//\"/\"\"}"
  printf '"%s"' "$s"
}

# csv_row <tool> <metric> <item> <threads> <repeat> <value> <unit>
# Appends one row for the current binary ($NAME). Rows without a value are skipped.
csv_row() {
  local tool="$1" metric="$2" item="$3" threads="$4" repeat="$5" value="$6" unit="$7"
  [[ -n "$value" ]] || return 0
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$RUN_ID" "$(csv_quote "$LABEL")" "$ARCH" "$NAME" "$tool" "$metric" \
    "$(csv_quote "$item")" "$threads" "$repeat" "$value" "$unit" >>"$RESULTS_CSV"
}

# add_rows <tool> <threads>: turn "metric|item|value|unit" lines from a parser into rows
add_rows() {
  local tool="$1" threads="$2" metric item value unit
  while IFS='|' read -r metric item value unit; do
    csv_row "$tool" "$metric" "$item" "$threads" "" "$value" "$unit"
  done
}

append_all_results() { # copy this run's rows into the file that holds every run
  [[ -f "$ALL_RESULTS_CSV" ]] || echo "$CSV_HEADER" >"$ALL_RESULTS_CSV"
  tail -n +2 "$RESULTS_CSV" >>"$ALL_RESULTS_CSV"
}
##################################################################################

##################################################################################
# =============================================================================
# PARSERS: raw tool output -> "metric|item|value|unit" lines
# Large counts are printed with %.0f, because Ubuntu's awk (mawk) would
# otherwise write them as 1.2e+10.
# =============================================================================
##################################################################################

parse_tok_s() { # last "achieved tok/s: X" or "forward tokens/s: X" line of a run's output
  awk '/achieved tok\/s|forward tokens\/s/ { v = $NF } END { print v }' "$1"
}

parse_time() { # /usr/bin/time -v
  awk -F': ' '
    /Maximum resident set size/     { print "max_rss_kb||" $2 "|kB" }
    /Percent of CPU this job got/   { v = $2; sub(/%/, "", v); print "cpu_percent||" v "|%" }
    /User time \(seconds\)/         { print "user_s||" $2 "|s" }
    /System time \(seconds\)/       { print "sys_s||" $2 "|s" }
    /Major \(requiring I\/O\) page/ { print "major_page_faults||" $2 "|count" }
    /Minor \(reclaiming a frame\)/  { print "minor_page_faults||" $2 "|count" }
    /Voluntary context switches/    { print "voluntary_ctx_switches||" $2 "|count" }
    /Involuntary context switches/  { print "involuntary_ctx_switches||" $2 "|count" }
  ' "$1"
}

parse_perf_stat() { # "26389.68 msec task-clock" -> task_clock_ms; "<not supported>" lines are skipped
  awk '
    $1 ~ /^[0-9][0-9.,]*$/ {
      v = $1; gsub(/,/, "", v)
      if ($2 == "msec")         { name = $3 "_ms"; unit = "ms" }
      else if ($2 == "seconds") { name = ($3 == "time" ? "elapsed" : $3) "_s"; unit = "s" }
      else                      { name = $2; unit = "count" }
      gsub(/[-\/:]/, "_", name)
      print name "||" v "|" unit
    }
  ' "$1"
}

parse_perf_report() { # "58.85%  cmd  dso  [.] matmul" -> self % per function, top TOP_N
  awk '
    $1 ~ /^[0-9.]+%$/ && ($4 == "[.]" || $4 == "[k]") {
      pct = $1; sub(/%/, "", pct)
      sym = $5; for (i = 6; i <= NF; i++) sym = sym " " $i
      if (sym ~ /^0x/) sym = ($3 ~ /^\[/ ? $3 : "[" $3 "]") # unnamed addresses: group by library
      self[sym] += pct
    }
    END { for (s in self) printf "self_pct|%s|%.2f|%%\n", s, self[s] }
  ' "$1" | sort -t'|' -k3,3gr | head -n "$TOP_N"
}

parse_gprof() { # flat profile rows: %time, cumulative s, self s, [calls, ms/call, ms/call,] name
  awk -v top="$TOP_N" '
    /^ *time +seconds +seconds/ { in_flat = 1; next } # column header of the flat profile
    in_flat && NF == 0          { exit }              # blank line ends it
    in_flat && n < top {
      n++
      print "self_pct|" $NF "|" $1 "|%"
      print "self_s|" $NF "|" $3 "|s"
      if (NF >= 7) print "calls|" $NF "|" $4 "|count"
    }
  ' "$1"
}

parse_callgrind() { # raw callgrind.out -> total + self instruction count (Ir) per function
  awk '
    /^fn=/ { # "fn=(12) matmul" names function 12; a later "fn=(12)" refers back to it
      s = substr($0, 4)
      if (match(s, /^\([0-9]+\)/)) {
        id = substr(s, 2, RLENGTH - 2)
        rest = substr(s, RLENGTH + 1); sub(/^ /, "", rest)
        if (rest != "") names[id] = rest
        cur = names[id]
      } else cur = s
      next
    }
    /^calls=/ { skip = 1; next } # the next cost line is a call (inclusive cost), not self cost
    /^[0-9+*-]/ { if (skip) skip = 0; else self[cur] += $2; next }
    /^(summary|totals):/ { total = $2 }
    END {
      printf "Ir_total||%.0f|instr\n", total
      for (f in self) printf "Ir|%s|%.0f|instr\n", f, self[f]
    }
  ' "$1" | sort -t'|' -k3,3gr | head -n "$((TOP_N + 1))" # Ir_total is the largest, so it sorts first
}

parse_cachegrind_totals() { # raw cachegrind.out "summary:" line -> event totals + miss rates
  awk '
    /^events:/  { for (i = 2; i <= NF; i++) ev[i - 1] = $i }
    /^summary:/ { for (i = 2; i <= NF; i++) tot[ev[i - 1]] = $i }
    END {
      for (e in tot) printf "%s||%.0f|count\n", e, tot[e]
      if (tot["Ir"] > 0) printf "I1_miss_rate||%.4f|%%\n", 100 * tot["I1mr"] / tot["Ir"]
      data = tot["Dr"] + tot["Dw"]
      if (data > 0) {
        printf "D1_miss_rate||%.4f|%%\n", 100 * (tot["D1mr"] + tot["D1mw"]) / data
        printf "LL_data_miss_rate||%.4f|%%\n", 100 * (tot["DLmr"] + tot["DLmw"]) / data
      }
    }
  ' "$1"
}

parse_cachegrind_functions() { # raw cachegrind.out -> L1 data misses (read + write) per function
  awk '
    /^events:/ { for (i = 2; i <= NF; i++) ev[i - 1] = $i; next }
    /^fn=/     { cur = substr($0, 4); next }
    /^[0-9]/ { # "<line> <one count per event>"
      for (i = 2; i <= NF; i++) if (ev[i - 1] == "D1mr" || ev[i - 1] == "D1mw") d1[cur] += $i
    }
    END { for (f in d1) if (d1[f] > 0) printf "D1_misses|%s|%.0f|count\n", f, d1[f] }
  ' "$1" | sort -t'|' -k3,3gr | head -n "$TOP_N"
}

parse_massif() { # peak heap; the mmap-ed model weights are not heap, so not included
  awk -F= '
    /^mem_heap_B=/       { if ($2 > heap) heap = $2 }
    /^mem_heap_extra_B=/ { if ($2 > extra) extra = $2 }
    END { print "peak_heap_bytes||" heap "|B"; print "peak_heap_extra_bytes||" extra "|B" }
  ' "$1"
}

# STEPS (each uses the globals NAME, BIN, OUT, ARGS, VG_ARGS set in MAIN)

# timed_run <threads> <output file>: run the binary once; sets WALL (s) and TOK (tok/s)
timed_run() {
  local threads="$1" logf="$2" start end
  start="$(date +%s.%N)"
  OMP_NUM_THREADS="$threads" "$BIN" "${ARGS[@]}" >"$logf" 2>&1 || return 1
  end="$(date +%s.%N)"
  WALL="$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.6f", e - s }')"
  TOK="$(parse_tok_s "$logf")"
}

# run_repeats <tool> <threads> <file prefix>: 1 warm-up + RUNS measured runs
# CSV: wall_s, tok_s and bandwidth per repeat. Files: runs/<prefix>warmup.txt, runs/<prefix>run_NN.txt
# bandwidth ~= model_bytes * tok/s: every weight is read once per token (KV cache and
# activations are small next to the weights, so this is an estimate)
run_repeats() {
  local tool="$1" threads="$2" prefix="$3" r logf model_bytes bw
  model_bytes="$(stat -c %s "$(model_for "$NAME")")"
  mkdir -p "$OUT/runs"
  timed_run "$threads" "$OUT/runs/${prefix}warmup.txt" || return 1
  for ((r = 1; r <= RUNS; r++)); do
    logf="$OUT/runs/${prefix}run_$(printf '%02d' "$r").txt"
    timed_run "$threads" "$logf" || return 1
    bw=""
    if [[ -n "$TOK" ]]; then
      bw="$(awk -v b="$model_bytes" -v t="$TOK" 'BEGIN { printf "%.4f", b * t / 1e9 }')"
    fi
    csv_row "$tool" wall_s "" "$threads" "$r" "$WALL" s
    csv_row "$tool" tok_s "" "$threads" "$r" "$TOK" tok/s
    csv_row "$tool" bandwidth "" "$threads" "$r" "$bw" GB/s
    printf '  threads=%s run %d: %s s, %s tok/s, %s GB/s\n' "$threads" "$r" "$WALL" "${TOK:-?}" "${bw:-?}"
  done
}

# CSV: this run's settings (tool=meta), so the rows can be interpreted without system.txt
run_meta() {
  local threads model
  threads="$(threads_for "$NAME")"
  model="$(model_for "$NAME")"
  csv_row meta steps "" "$threads" "" "$STEPS" tokens
  csv_row meta valgrind_steps "" "$threads" "" "$VALGRIND_STEPS" tokens
  csv_row meta runs "" "$threads" "" "$RUNS" count
  csv_row meta seed "" "$threads" "" "$SEED" ""
  csv_row meta model_bytes "$(basename "$model")" "$threads" "" "$(stat -c %s "$model")" B
}

# CSV tool=timing. Files: runs/, output.txt (the generated text of run 1)
run_timing() {
  run_repeats timing "$(threads_for "$NAME")" "" &&
    cp "$OUT/runs/run_01.txt" "$OUT/output.txt"
}

# CSV tool=thread_sweep: timing for each thread count. Files: runs/sweep_t<N>_*
run_thread_sweep() {
  local t list
  IFS=',' read -ra list <<<"$THREADS_SWEEP"
  for t in "${list[@]}"; do
    run_repeats thread_sweep "$t" "sweep_t${t}_" || return 1
  done
}

# CSV tool=time. File: time.txt
run_time() {
  /usr/bin/time -v -o "$OUT/time.txt" "$BIN" "${ARGS[@]}" >/dev/null &&
    parse_time "$OUT/time.txt" | add_rows time "$(threads_for "$NAME")"
}

# CSV tool=gprof. Files: gmon.out, gprof.txt (uses the -pg build)
run_gprof() {
  local gbin="$GPROF_BUILD_DIR/$NAME"
  (cd "$OUT" && "$gbin" "${ARGS[@]}" >/dev/null) && # writes gmon.out to the current dir
    gprof -b "$gbin" "$OUT/gmon.out" >"$OUT/gprof.txt" &&
    parse_gprof "$OUT/gprof.txt" | add_rows gprof "$(threads_for "$NAME")"
}

# CSV tool=perf_stat. File: perf_stat.txt
run_perf_stat() {
  perf stat -o "$OUT/perf_stat.txt" \
    -e task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions,cache-references,cache-misses,branches,branch-misses \
    -- "$BIN" "${ARGS[@]}" >/dev/null &&
    parse_perf_stat "$OUT/perf_stat.txt" | add_rows perf_stat "$(threads_for "$NAME")"
}

# CSV tool=perf_report. Files: perf.data, perf_report.txt
run_perf_record() {
  perf record -q -F 999 -e cpu-clock -g -o "$OUT/perf.data" -- "$BIN" "${ARGS[@]}" >/dev/null &&
    perf report -i "$OUT/perf.data" --stdio --no-children >"$OUT/perf_report.txt" 2>/dev/null &&
    parse_perf_report "$OUT/perf_report.txt" | add_rows perf_report "$(threads_for "$NAME")"
}

# CSV tool=callgrind. Files: callgrind.out, callgrind.txt, callgrind.log
run_callgrind() {
  valgrind --tool=callgrind --callgrind-out-file="$OUT/callgrind.out" \
    "$BIN" "${VG_ARGS[@]}" >/dev/null 2>"$OUT/callgrind.log" &&
    callgrind_annotate "$OUT/callgrind.out" >"$OUT/callgrind.txt" &&
    parse_callgrind "$OUT/callgrind.out" | add_rows callgrind 1
}

# CSV tool=cachegrind. Files: cachegrind.out, cachegrind.txt, cachegrind.log
run_cachegrind() {
  valgrind --tool=cachegrind --cache-sim=yes "${CACHEGRIND_CACHE[@]}" \
    --cachegrind-out-file="$OUT/cachegrind.out" \
    "$BIN" "${VG_ARGS[@]}" >/dev/null 2>"$OUT/cachegrind.log" &&
    cg_annotate "$OUT/cachegrind.out" >"$OUT/cachegrind.txt" &&
    {
      parse_cachegrind_totals "$OUT/cachegrind.out"
      parse_cachegrind_functions "$OUT/cachegrind.out"
    } | add_rows cachegrind 1
}

# CSV tool=massif. Files: massif.out, massif.txt, massif.log
run_massif() {
  valgrind --tool=massif --massif-out-file="$OUT/massif.out" \
    "$BIN" "${VG_ARGS[@]}" >/dev/null 2>"$OUT/massif.log" &&
    ms_print "$OUT/massif.out" >"$OUT/massif.txt" &&
    parse_massif "$OUT/massif.out" | add_rows massif 1
}

##################################################################################
# MAIN
##################################################################################
check_inputs
build
mkdir -p "$LOG_ROOT"
echo "$CSV_HEADER" >"$RESULTS_CSV"
system_info

for NAME in "${BINARIES[@]}"; do
  BIN="$BUILD_DIR/$NAME"
  OUT="$LOG_ROOT/$NAME"
  mkdir -p "$OUT"
  mapfile -t ARGS < <(args_for "$NAME" "$STEPS")
  mapfile -t VG_ARGS < <(args_for "$NAME" "$VALGRIND_STEPS")

  # Every binary
  run_meta
  step "$NAME: timing" run_timing
  step "$NAME: time -v" run_time
  step "$NAME: perf stat" run_perf_stat
  step "$NAME: perf record" run_perf_record

  # Binary-specific
  case "$NAME" in
  run_fp32_baseline | run_int8_baseline)
    step "$NAME: gprof" run_gprof
    step "$NAME: callgrind" run_callgrind
    step "$NAME: cachegrind" run_cachegrind
    step "$NAME: massif" run_massif
    ;;
  run_fp32_simd | run_fp32_simd_4row)
    step "$NAME: gprof" run_gprof
    step "$NAME: callgrind" run_callgrind
    step "$NAME: cachegrind" run_cachegrind
    ;;
  run_fp32_openmp | run_fp32_openmp_simd)
    # gprof only sees the main thread, and valgrind runs threads one at a time,
    # so neither says much about a multi-threaded build
    step "$NAME: thread sweep" run_thread_sweep
    ;;
  esac
done

append_all_results

log "Results in ${LOG_ROOT#"$ROOT"/} ($(($(wc -l <"$RESULTS_CSV") - 1)) rows in results.csv)"
if ((${#FAILURES[@]})); then
  printf 'Failed steps:\n'
  printf '  - %s\n' "${FAILURES[@]}"
  exit 1
fi
