#!/usr/bin/env bash
# Save as scripts/agreement.sh. Run INSIDE Docker: bash scripts/agreement.sh
# Optional: bash scripts/agreement.sh 100
# Optional: BUILD_DIR=build-arm64 THREADS=4 bash scripts/agreement.sh 10
# Input defaults to data/TinyStories-valid.txt; override with STORIES_FILE=...
# Uses already-built, agreement-enabled C runners. No Python required.
set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: bash scripts/agreement.sh [number_of_stories=10]'
    echo 'Environment: BUILD_DIR=build-arm64 THREADS=4 STORIES_FILE=data/TinyStories-valid.txt'
    echo 'Skips the initial (possibly partial) story segment, removes separators/CRLF,'
    echo 'then evaluates the requested number of complete stories as continuous text.'
    echo 'Uses existing binaries; rebuild after changing C sources.'
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ ! -f "$ROOT/CMakeLists.txt" ]]; then
    echo 'Save this file in your TinyLLM project scripts/ folder.' >&2
    exit 1
fi
cd "$ROOT"

STORY_COUNT="${1:-10}"
BUILD_DIR="${BUILD_DIR:-build-arm64}"
THREADS="${THREADS:-4}"
STORIES_FILE="${STORIES_FILE:-data/TinyStories-valid.txt}"
[[ $# -le 1 && "$STORY_COUNT" =~ ^[1-9][0-9]*$ && ${#STORY_COUNT} -le 6 ]] || {
    echo 'Provide one story count between 1 and 999999, e.g. bash scripts/agreement.sh 10' >&2
    exit 1
}
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || { echo 'THREADS must be a positive integer.' >&2; exit 1; }
[[ -s "$STORIES_FILE" ]] || { echo "Missing/empty input: $STORIES_FILE" >&2; exit 1; }

runners=(run_fp32_baseline run_int8_baseline run_fp16_baseline
         run_fp16_baseline_native run_fp32_simd run_fp32_openmp run_fp32_openmp_neon)
models=(stories110M.bin stories110M_int8.bin stories110M_fp16.bin
        stories110M_fp16.bin stories110M.bin stories110M.bin stories110M.bin)

# Check all prerequisites before starting expensive model runs.
for runner in "${runners[@]}" compare_next_tokens; do
    [[ -x "$BUILD_DIR/$runner" ]] || {
        echo "Missing executable: $BUILD_DIR/$runner. Build this target first." >&2
        exit 1
    }
done
for model in stories110M.bin stories110M_int8.bin stories110M_fp16.bin tokenizer.bin; do
    [[ -s "models/$model" ]] || { echo "Missing/empty model file: models/$model" >&2; exit 1; }
done

mkdir -p logs/agreement
RESULTS="$(mktemp -d "logs/agreement/$(date +%Y%m%d-%H%M%S)-XXXXXX")"
echo "Results directory: $ROOT/$RESULTS"
trap 'echo "Test failed or stopped; partial files are in: $ROOT/$RESULTS" >&2' ERR

# Do not score a final unterminated story. Nothing is run unless exactly N
# nonempty, delimiter-terminated stories were found after the first segment.
TEXT="$RESULTS/evaluation.txt"
if ! awk -v wanted="$STORY_COUNT" '
    { sub(/\r$/, "") }
    $0 == "<|endoftext|>" {
        if (!started) { started=1; next }
        if (nonempty) {
            printf "%s\n", story
            count++
        }
        story=""; nonempty=0
        if (count >= wanted) exit
        next
    }
    started {
        story=story $0 "\n"
        if ($0 ~ /[^[:space:]]/) nonempty=1
    }
    END { if (count != wanted) exit 2 }
' "$STORIES_FILE" > "$TEXT"; then
    echo "Could not extract $STORY_COUNT complete stories after the first segment." >&2
    echo 'Check the input uses <|endoftext|> separator lines, or request fewer stories.' >&2
    exit 1
fi

{
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'source=%s\nstory_count=%s\nskip_initial_segment=yes\n' "$STORIES_FILE" "$STORY_COUNT"
    printf 'build_dir=%s\nopenmp_threads=%s\ncontext_policy=continuous_text\n' "$BUILD_DIR" "$THREADS"
    printf 'architecture=%s\n' "$(uname -m)"
    if command -v sha256sum >/dev/null; then sha256sum "$TEXT"; fi
} > "$RESULTS/settings.txt"

for i in "${!runners[@]}"; do
    runner="${runners[$i]}"
    threads=1
    case "$runner" in *openmp*) threads="$THREADS" ;; esac
    echo "Evaluating $runner (OMP_NUM_THREADS=$threads)..."
    if ! OMP_NUM_THREADS="$threads" "$BUILD_DIR/$runner" "models/${models[$i]}" \
        -z models/tokenizer.bin -m agreement -f "$TEXT" \
        -o "$RESULTS/$runner.csv" > "$RESULTS/$runner.log" 2>&1; then
        cat "$RESULTS/$runner.log" >&2
        echo "$runner failed; comparison was not run. Results: $ROOT/$RESULTS" >&2
        exit 1
    fi
    cat "$RESULTS/$runner.log"
done

candidates=()
for runner in "${runners[@]:1}"; do candidates+=("$RESULTS/$runner.csv"); done
if ! "$BUILD_DIR/compare_next_tokens" "$RESULTS/run_fp32_baseline.csv" \
    "${candidates[@]}" > "$RESULTS/summary.csv.tmp"; then
    echo 'Comparison failed. Do not use the partial summary.csv.tmp.' >&2
    exit 1
fi
mv "$RESULTS/summary.csv.tmp" "$RESULTS/summary.csv"
echo
echo 'Next-token agreement with FP32:'
cat "$RESULTS/summary.csv"
echo
echo "Completed. Results: $ROOT/$RESULTS"