/* Shared teacher-forced next-token trace for every model variant. */
#include "agreement.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int agreement_file(void *model, agreement_forward_fn forward,
                   int vocab_size, int seq_len, Tokenizer *tokenizer,
                   const char *text_path, const char *output_path)
{
    FILE *input = NULL, *output = NULL;
    char *text = NULL;
    int *tokens = NULL;
    int status = 1, n_tokens = 0;
    size_t scored = 0;
    if (!forward || !tokenizer || !text_path || !output_path ||
        vocab_size <= 0 || seq_len < 2) {
        fprintf(stderr, "agreement: invalid arguments or context length\n");
        return 1;
    }
    if (strcmp(text_path, output_path) == 0) {
        fprintf(stderr, "agreement: text and output must be different files\n");
        return 1;
    }
    input = fopen(text_path, "rb");
    if (!input || fseek(input, 0, SEEK_END) != 0) {
        fprintf(stderr, "agreement: cannot open/seek text file %s\n", text_path);
        goto cleanup;
    }
    long length = ftell(input);
    if (length < 0 || length > INT_MAX - 3 ||
        (uintmax_t)length + 3 > SIZE_MAX / sizeof(int) ||
        fseek(input, 0, SEEK_SET) != 0) {
        fprintf(stderr, "agreement: unsupported text size\n");
        goto cleanup;
    }
    text = malloc((size_t)length + 1);
    tokens = malloc(((size_t)length + 3) * sizeof(int));
    if (!text || !tokens || fread(text, 1, (size_t)length, input) != (size_t)length) {
        fprintf(stderr, "agreement: text read/allocation failed\n");
        goto cleanup;
    }
    if (memchr(text, '\0', (size_t)length)) {
        fprintf(stderr, "agreement: input must be text without embedded NUL bytes\n");
        goto cleanup;
    }
    text[length] = '\0';
    fclose(input); input = NULL;
    encode(tokenizer, text, 1, 0, tokens, &n_tokens);
    if (n_tokens < 2) {
        fprintf(stderr, "agreement: need at least one next-token prediction\n");
        goto cleanup;
    }
    for (int i = 0; i < n_tokens; ++i) {
        if (tokens[i] < 0 || tokens[i] >= vocab_size) {
            fprintf(stderr, "agreement: invalid encoded token at index %d\n", i);
            goto cleanup;
        }
    }
    output = fopen(output_path, "wb");
    if (!output) {
        fprintf(stderr, "agreement: cannot create %s (create its parent folder first)\n", output_path);
        goto cleanup;
    }
    fprintf(output, "# agreement_format=1\n# vocab_size=%d\n# seq_len=%d\n# token_count=%d\n",
            vocab_size, seq_len, n_tokens);
    fprintf(output, "index,chunk_start,pos,input_token,target_token,predicted_token\n");
    /* Match perplexity.c: reset positions at context boundaries and overlap
     * one token. All n_tokens-1 next-token positions are visited exactly once.
     */
    for (size_t start = 0; start < (size_t)n_tokens - 1; start += (size_t)seq_len - 1) {
        size_t len = (size_t)n_tokens - start;
        if (len > (size_t)seq_len) len = (size_t)seq_len;
        for (size_t pos = 0; pos + 1 < len; ++pos) {
            float *logits = forward(model, tokens[start + pos], (int)pos);
            if (!logits) { fprintf(stderr, "agreement: null logits\n"); goto cleanup; }
            int predicted = 0;
            for (int id = 0; id < vocab_size; ++id) {
                if (!isfinite(logits[id])) {
                    fprintf(stderr, "agreement: nonfinite logit at position %zu, token %d\n", start + pos, id);
                    goto cleanup;
                }
                /* Strict > selects the lowest token ID on an exact tie. */
                if (logits[id] > logits[predicted]) predicted = id;
            }
            if (fprintf(output, "%zu,%zu,%zu,%d,%d,%d\n", start + pos, start, pos,
                        tokens[start + pos], tokens[start + pos + 1], predicted) < 0) {
                fprintf(stderr, "agreement: trace write failed\n");
                goto cleanup;
            }
            ++scored;
        }
    }
    if (ferror(output)) { fprintf(stderr, "agreement: trace write failed\n"); goto cleanup; }
    if (fclose(output) != 0) {
        output = NULL;
        fprintf(stderr, "agreement: trace close failed\n");
        goto cleanup;
    }
    output = NULL;
    fprintf(stderr, "agreement positions: %zu; saved: %s\n", scored, output_path);
    status = 0;
cleanup:
    if (input) fclose(input);
    if (output) fclose(output);
    free(tokens);
    free(text);
    return status;
}
