/**
 * @file perplexity.h
 * @brief Perplexity (PPL) of a model on a text file, for measuring accuracy loss (e.g. FP32 vs INT8).
 *
 * PPL = exp(mean negative log-likelihood of each real next token), using teacher forcing:
 * the model is fed the real text, never its own samples, so the result does not depend on
 * temperature/top-p. Lower is better; identical weights + math give identical PPL.
 *
 * Model-agnostic: the caller passes its own forward() through a function pointer, so the same
 * code works for any Transformer struct (FP32, INT8, ...).
 **/

#ifndef PERPLEXITY_H
#define PERPLEXITY_H

#include "tokenizer.h"

// Signature every model must provide: logits[vocab_size] for `token` at position `pos`
typedef float *(*forward_fn)(void *model, int token, int pos);

typedef struct
{
  double nll_sum;// sum of -log p(real next token), in nats
  long n_scored;// number of predicted tokens
  double perplexity;// exp(nll_sum / n_scored)
} PerplexityResult;

// log p(target) from raw logits; numerically stable (max trick, double sums), logits unchanged
double log_prob(const float *logits, int target, int vocab_size);

// PPL over an already-tokenized sequence. Sequences longer than seq_len are split into chunks,
// each restarting at pos 0 (the KV cache is overwritten by position, so no reset is needed).
PerplexityResult
  perplexity_tokens(void *model, forward_fn forward, int vocab_size, int seq_len, const int *tokens, int n_tokens);

// Read a text file, tokenize it (with BOS), compute PPL and print a summary to stderr
PerplexityResult perplexity_file(void *model,
  forward_fn forward,
  int vocab_size,
  int seq_len,
  Tokenizer *tokenizer,
  const char *text_path);

#endif// PERPLEXITY_H
