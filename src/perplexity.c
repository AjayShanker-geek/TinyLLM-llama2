/**
 * @file perplexity.c
 * @brief Perplexity of a model on a text file. See perplexity.h.
 **/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "perplexity.h"

double log_prob(const float *logits, int target, int vocab_size)
{
  // log softmax(logits)[target] = (l_target - max) - log(sum_i exp(l_i - max))
  // subtracting max keeps exp() from overflowing; double avoids losing precision over 32k terms
  float max_logit = logits[0];
  for (int i = 1; i < vocab_size; i++) {
    if (logits[i] > max_logit) { max_logit = logits[i]; }
  }
  double sum = 0.0;
  for (int i = 0; i < vocab_size; i++) { sum += exp((double)(logits[i] - max_logit)); }
  return (double)(logits[target] - max_logit) - log(sum);
}

PerplexityResult
  perplexity_tokens(void *model, forward_fn forward, int vocab_size, int seq_len, const int *tokens, int n_tokens)
{
  PerplexityResult r = { 0 };

  // each chunk feeds up to seq_len tokens and scores up to seq_len - 1 next-token predictions;
  // the last target of one chunk becomes the first input of the next, so every token is scored once
  for (int start = 0; start < n_tokens - 1; start += seq_len - 1) {
    int len = n_tokens - start;
    if (len > seq_len) { len = seq_len; }

    for (int pos = 0; pos < len - 1; pos++) {
      float *logits = forward(model, tokens[start + pos], pos);
      r.nll_sum -= log_prob(logits, tokens[start + pos + 1], vocab_size);
      r.n_scored++;
    }
  }

  r.perplexity = r.n_scored > 0 ? exp(r.nll_sum / (double)r.n_scored) : NAN;
  return r;
}

// Read the whole file into a NUL-terminated buffer (caller frees)
static char *read_text_file(const char *path)
{
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "couldn't open %s\n", path);
    exit(EXIT_FAILURE);
  }
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);

  char *text = malloc((size_t)size + 1);
  if (!text || fread(text, 1, (size_t)size, f) != (size_t)size) {
    fprintf(stderr, "failed to read %s\n", path);
    exit(EXIT_FAILURE);
  }
  text[size] = '\0';
  fclose(f);
  return text;
}

PerplexityResult perplexity_file(void *model,
  forward_fn forward,
  int vocab_size,
  int seq_len,
  Tokenizer *tokenizer,
  const char *text_path)
{
  char *text = read_text_file(text_path);

  // +3 for BOS, dummy prefix and EOS; BPE only ever merges, so tokens <= bytes
  int *tokens = malloc((strlen(text) + 3) * sizeof(int));
  int n_tokens = 0;
  encode(tokenizer, text, 1, 0, tokens, &n_tokens);

  PerplexityResult r = perplexity_tokens(model, forward, vocab_size, seq_len, tokens, n_tokens);

  fprintf(stderr, "ppl tokens: %ld\n", r.n_scored);
  fprintf(stderr, "ppl mean nll: %f\n", r.n_scored > 0 ? r.nll_sum / (double)r.n_scored : NAN);
  fprintf(stderr, "ppl perplexity: %f\n", r.perplexity);

  free(tokens);
  free(text);
  return r;
}
