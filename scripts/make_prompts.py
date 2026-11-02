"""Build the fixed benchmark prompt set from the TinyStories validation split.

Shuffles the validation stories with a fixed seed, then takes the first sentence
of each, skipping duplicates, until N unique prompts are collected. A random
sample keeps the mix of story openings close to the validation set's (most, but
not all, start with "Once upon a time"). The first entry of the file is a
cut-off story fragment, so it is skipped.

Usage (from the project root):
    python3 scripts/make_prompts.py                     # 20 prompts, seed 42 -> data/prompts.txt
    python3 scripts/make_prompts.py -n 20 -s 42 -o data/prompts.txt
"""

import argparse
import random
import re

SEPARATOR = "<|endoftext|>"
# First sentence: up to the first . ! or ? (plus a closing quote if there is one)
FIRST_SENTENCE = re.compile(r'^(.+?[.!?]["”]?)(\s|$)', re.S)


def first_sentence(story):
    text = " ".join(story.split())  # collapse newlines and repeated spaces
    m = FIRST_SENTENCE.match(text)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input", default="data/TinyStories-valid.txt")
    ap.add_argument("-o", "--output", default="data/prompts.txt")
    ap.add_argument("-n", "--count", type=int, default=20)
    ap.add_argument("-s", "--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        stories = f.read().split(SEPARATOR)[1:]  # [0] is a cut-off fragment

    # fixed seed: the same prompts on every machine and every rerun
    random.Random(args.seed).shuffle(stories)

    prompts, seen = [], set()
    for story in stories:
        sentence = first_sentence(story)
        if sentence and sentence not in seen:
            seen.add(sentence)
            prompts.append(sentence)
            if len(prompts) == args.count:
                break

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(prompts) + "\n")
    print(f"wrote {len(prompts)} prompts to {args.output}")


if __name__ == "__main__":
    main()
