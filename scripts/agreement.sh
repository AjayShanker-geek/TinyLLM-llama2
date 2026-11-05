#!/usr/bin/env bash
# Save as scripts/agreement.sh. Run INSIDE Docker: bash scripts/agreement.sh
# Optional: bash scripts/agreement.sh 3
# Optional: BUILD_DIR=build-arm64 THREADS=4 bash scripts/agreement.sh 3
# Default: extract 3 stories from data/TinyStories-valid.txt.
# To use your own data/story1.txt etc: STORY_DIR=data bash scripts/agreement.sh 3
# Uses already-built, agreement-enabled C runners. No Python required.
set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: bash scripts/agreement.sh [number_of_stories=3]'
    echo 'Environment: BUILD_DIR=build-arm64 THREADS=4 STORIES_FILE=data/TinyStories-valid.txt'
    echo 'Skips the initial (possibly partial) story segment, removes separators/CRLF,'
    echo 'then evaluates each story in a fresh process with BOS and fresh context.'
    echo 'To use existing story1.txt etc: STORY_DIR=data bash scripts/agreement.sh 3'
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

STORY_COUNT="${1:-3}"
STORY_DIR="${STORY_DIR:-}"
BUILD_DIR="${BUILD_DIR:-build-arm64}"
THREADS="${THREADS:-4}"
STORIES_FILE="${STORIES_FILE:-data/TinyStories-valid.txt}"
[[ $# -le 1 && "$STORY_COUNT" =~ ^[1-9][0-9]*$ && ${#STORY_COUNT} -le 6 ]] || {
    echo 'Provide one story count between 1 and 999999, e.g. bash scripts/agreement.sh 3' >&2
    exit 1
}
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || { echo 'THREADS must be a positive integer.' >&2; exit 1; }
if [[ -n "$STORY_DIR" ]]; then
    for ((story=1; story<=STORY_COUNT; story++)); do
        [[ -s "$STORY_DIR/story$story.txt" ]] || {
            echo "Missing/empty input: $STORY_DIR/story$story.txt" >&2; exit 1;
        }
    done
else
    [[ -s "$STORIES_FILE" ]] || { echo "Missing/empty input: $STORIES_FILE" >&2; exit 1; }
fi

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

# Snapshot each story so all runners read exactly the same bytes.
mkdir -p "$RESULTS/stories"
if [[ -n "$STORY_DIR" ]]; then
    for ((story=1; story<=STORY_COUNT; story++)); do
        sed 's/\r$//' "$STORY_DIR/story$story.txt" > "$RESULTS/stories/story$story.txt"
        if ! grep -q '[^[:space:]]' "$RESULTS/stories/story$story.txt"; then
            echo "story$story.txt has no text." >&2; exit 1
        fi
    done
else
    # Skip the initial partial segment in the uploaded TinyStories file.
    # Only delimiter-terminated, nonempty stories are selected.
    if ! awk -v wanted="$STORY_COUNT" -v directory="$RESULTS/stories" '
        { sub(/\r$/, "") }
        $0 == "<|endoftext|>" {
            if (!started) { started=1; next }
            if (nonempty) {
                count++
                path=directory "/story" count ".txt"
                printf "%s", story > path
                close(path)
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
    ' "$STORIES_FILE"; then
        echo "Could not extract $STORY_COUNT complete stories after the first segment." >&2
        echo 'Request fewer stories, or use STORY_DIR=data with story1.txt, story2.txt, etc.' >&2
        exit 1
    fi
fi

{
    printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ -n "$STORY_DIR" ]]; then
        printf 'source_story_dir=%s\nskip_initial_segment=no\n' "$STORY_DIR"
    else
        printf 'source=%s\nskip_initial_segment=yes\n' "$STORIES_FILE"
    fi
    printf 'story_count=%s\nbuild_dir=%s\nopenmp_threads=%s\n' "$STORY_COUNT" "$BUILD_DIR" "$THREADS"
    printf 'context_policy=fresh_process_and_BOS_per_story; existing_chunking_within_story\n'
    printf 'architecture=%s\n' "$(uname -m)"
    if command -v sha256sum >/dev/null; then
        for ((story=1; story<=STORY_COUNT; story++)); do
            sha256sum "$RESULTS/stories/story$story.txt"
        done
    fi
} > "$RESULTS/settings.txt"

summaries=()
for ((story=1; story<=STORY_COUNT; story++)); do
    name="story$story"
    TEXT="$RESULTS/stories/$name.txt"
    directory="$RESULTS/$name"
    mkdir -p "$directory"
    echo
    echo "=== $name ($story/$STORY_COUNT) ==="
    for i in "${!runners[@]}"; do
        runner="${runners[$i]}"
        threads=1
        case "$runner" in *openmp*) threads="$THREADS" ;; esac
        echo "Evaluating $runner on $name (OMP_NUM_THREADS=$threads)..."
        # A separate invocation loads a fresh model and tokenizer for EACH story.
        # agreement.c inserts BOS; prior stories cannot enter this process's cache.
        if ! OMP_NUM_THREADS="$threads" "$BUILD_DIR/$runner" "models/${models[$i]}" \
            -z models/tokenizer.bin -m agreement -f "$TEXT" \
            -o "$directory/$runner.csv" > "$directory/$runner.log" 2>&1; then
            cat "$directory/$runner.log" >&2
            echo "$runner failed on $name. Results: $ROOT/$RESULTS" >&2
            exit 1
        fi
        cat "$directory/$runner.log"
    done

    candidates=()
    for runner in "${runners[@]:1}"; do candidates+=("$directory/$runner.csv"); done
    if ! "$BUILD_DIR/compare_next_tokens" "$directory/run_fp32_baseline.csv" \
        "${candidates[@]}" > "$directory/summary.csv.tmp"; then
        echo "Comparison failed on $name; do not use partial summary.csv.tmp." >&2
        exit 1
    fi
    mv "$directory/summary.csv.tmp" "$directory/summary.csv"
    summaries+=("$directory/summary.csv")
    cat "$directory/summary.csv"
done

# The comparator emits candidates in argument order. Sum integer counts for
# each candidate across stories; do NOT average the story percentages.
# Numeric fields are the last four columns, even if a CSV path contains commas.
awk -F, -v variants="${runners[*]:1}" -v stories="$STORY_COUNT" '
    BEGIN { n=split(variants, names, " ") }
    FNR == 1 { next }
    {
        id=FNR-1
        positions[id]+=$(NF-3)
        matches[id]+=$(NF-2)
        mismatches[id]+=$(NF-1)
        rows[id]++
    }
    END {
        for (i=1; i<=n; i++) {
            if (rows[i] != stories || positions[i] <= 0) exit 2
        }
        print "variant,positions,matches,mismatches,agreement_percent"
        for (i=1; i<=n; i++)
            printf "%s,%.0f,%.0f,%.0f,%.6f\n", names[i], positions[i], matches[i], mismatches[i], 100*matches[i]/positions[i]
    }
' "${summaries[@]}" > "$RESULTS/summary.csv.tmp"
mv "$RESULTS/summary.csv.tmp" "$RESULTS/summary.csv"
echo
echo 'Overall next-token agreement with FP32 (all selected stories):'
cat "$RESULTS/summary.csv"
echo
echo "Completed. Results: $ROOT/$RESULTS"