#ifndef NEXT_TOKEN_AGREEMENT_H
#define NEXT_TOKEN_AGREEMENT_H

#include "tokenizer.h"

typedef float *(*agreement_forward_fn)(void *model, int token, int pos);

/* Teacher-force the input text and save the argmax prediction at each position.
 * Returns 0 on success; each runner retains ownership of its model/tokenizer.
 */
int agreement_file(void *model, agreement_forward_fn forward,
                   int vocab_size, int seq_len, Tokenizer *tokenizer,
                   const char *text_path, const char *output_path);

#endif
