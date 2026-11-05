/**
 * @file run_int8_baseline.c
 * @brief This file contains the implementation of the int8 model for inference.
 * @author TODO: Update author name
 **/

#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif
#include "run_int8_baseline.h"
/* Inference for Llama-2 Transformer model in pure C, int8 quantized forward pass. */

#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "perplexity.h"
#include "tokenizer.h"

// ----------------------------------------------------------------------------
// Globals
int GS = 0;// group size global for quantization of the weights

// ----------------------------------------------------------------------------
// Transformer model

typedef struct
{
  int dim;// transformer dimension
  int hidden_dim;// for ffn layers
  int n_layers;// number of layers
  int n_heads;// number of query heads
  int n_kv_heads;// number of key/value heads (can be < query heads because of multiquery)
  int vocab_size;// vocabulary size, usually 256 (byte-level)
  int seq_len;// max sequence length
} Config;

typedef struct
{
  // token embedding table
  QuantizedTensor *q_tokens;// (vocab_size, dim)
  float *token_embedding_table;// same, but dequantized

  // weights for rmsnorms
  float *rms_att_weight;// (layer, dim) rmsnorm weights
  float *rms_ffn_weight;// (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  QuantizedTensor *wq;// (layer, dim, n_heads * head_size)
  QuantizedTensor *wk;// (layer, dim, n_kv_heads * head_size)
  QuantizedTensor *wv;// (layer, dim, n_kv_heads * head_size)
  QuantizedTensor *wo;// (layer, n_heads * head_size, dim)
  // weights for ffn
  QuantizedTensor *w1;// (layer, hidden_dim, dim)
  QuantizedTensor *w2;// (layer, dim, hidden_dim)
  QuantizedTensor *w3;// (layer, hidden_dim, dim)
  // final rmsnorm
  float *rms_final_weight;// (dim,)
  // (optional) classifier weights for the logits, on the last layer
  QuantizedTensor *wcls;
} TransformerWeights;

typedef struct
{
  // current wave of activations
  float *x;// activation at current time stamp (dim,)
  float *xb;// same, but inside a residual branch (dim,)
  float *xb2;// an additional buffer just for convenience (dim,)
  float *hb;// buffer for hidden dimension in the ffn (hidden_dim,)
  float *hb2;// buffer for hidden dimension in the ffn (hidden_dim,)
  QuantizedTensor xq;// quantized x (dim,)
  QuantizedTensor hq;// quantized hb (hidden_dim,)
  float *q;// query (dim,)
  float *k;// key (dim,)
  float *v;// value (dim,)
  float *att;// buffer for scores/attention values (n_heads, seq_len)
  float *logits;// output logits
  // kv cache
  float *key_cache;// (layer, seq_len, dim)
  float *value_cache;// (layer, seq_len, dim)
} RunState;

typedef struct
{
  Config config;// the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights;// the weights of the model
  RunState state;// buffers for the "wave" of activations in the forward pass
  // some more state needed to properly clean up the memory mapping (sigh)
  int fd;// file descriptor for memory mapping
  float *data;// memory mapped data pointer
  size_t file_size;// size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState *s, Config *p)
{
  // we calloc instead of malloc to keep valgrind happy
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  s->x = calloc(p->dim, sizeof(float));
  s->xb = calloc(p->dim, sizeof(float));
  s->xb2 = calloc(p->dim, sizeof(float));
  s->hb = calloc(p->hidden_dim, sizeof(float));
  s->hb2 = calloc(p->hidden_dim, sizeof(float));
  s->xq = (QuantizedTensor){ .q = calloc(p->dim, sizeof(int8_t)), .s = calloc((p->dim + GS - 1) / GS, sizeof(float)) };
  s->hq = (QuantizedTensor){ .q = calloc(p->hidden_dim, sizeof(int8_t)),
    .s = calloc((p->hidden_dim + GS - 1) / GS, sizeof(float)) };
  s->q = calloc(p->dim, sizeof(float));
  s->k = calloc(kv_dim, sizeof(float));
  s->v = calloc(kv_dim, sizeof(float));
  s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
  s->logits = calloc(p->vocab_size, sizeof(float));
  s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
  s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
  // ensure all mallocs went fine
  if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v || !s->att || !s->logits
      || !s->key_cache || !s->value_cache || !s->xq.q || !s->xq.s || !s->hq.q || !s->hq.s) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
}

void free_run_state(RunState *s)
{
  free(s->x);
  free(s->xb);
  free(s->xb2);
  free(s->hb);
  free(s->hb2);
  free(s->xq.q);
  free(s->xq.s);
  free(s->hq.q);
  free(s->hq.s);
  free(s->q);
  free(s->k);
  free(s->v);
  free(s->att);
  free(s->logits);
  free(s->key_cache);
  free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float *x, int n)
{
  for (int i = 0; i < n; i++) { x[i] = qx->q[i] * qx->s[i / GS]; }
}

/* One scale per group. Handles zero groups and a final partial group.
 * Model files still require whole groups in each matrix row (see loader).
 */
void quantize(QuantizedTensor *qx, float *x, int n)
{
  const int groups = (n + GS - 1) / GS;
  for (int g = 0; g < groups; g++) {
    int start = g * GS;
    int count = n - start < GS ? n - start : GS;
    float maximum = 0.0f;
    for (int k = 0; k < count; k++) {
      float value = fabsf(x[start + k]);
      if (!isfinite(value)) {
        fprintf(stderr, "Cannot quantize a non-finite activation\n");
        exit(EXIT_FAILURE);
      }
      if (value > maximum) maximum = value;
    }
    float scale = maximum / 127.0f;
    qx->s[g] = scale;
    for (int k = 0; k < count; k++) {
      /* A zero group must not divide by zero. */
      float value = scale == 0.0f ? 0.0f : roundf(x[start + k] / scale);
      if (value > 127.0f) value = 127.0f;
      if (value < -127.0f) value = -127.0f;
      qx->q[start + k] = (int8_t)value;
    }
  }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each)
{
  void *p = *ptr;
  QuantizedTensor *res = malloc((size_t)n * sizeof(QuantizedTensor));
  if (!res) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < n; i++) {
    /* map quantized int8 values*/
    res[i].q = (int8_t *)p;
    p = (int8_t *)p + size_each;
    /* map scale factors */
    res[i].s = (float *)p;
    p = (float *)p + size_each / GS;
  }
  *ptr = p;// advance ptr to current position
  return res;
}

void memory_map_weights(TransformerWeights *w, Config *p, void *ptr, uint8_t shared_classifier)
{
  int head_size = p->dim / p->n_heads;
  // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
  float *fptr = (float *)ptr;// cast our pointer to float*
  w->rms_att_weight = fptr;
  fptr += p->n_layers * p->dim;
  w->rms_ffn_weight = fptr;
  fptr += p->n_layers * p->dim;
  w->rms_final_weight = fptr;
  fptr += p->dim;

  // now read all the quantized weights
  ptr = (void *)fptr;// now cast the pointer back to void*
  w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
  // dequantize token embedding table
  w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
  if (!w->token_embedding_table) {
    fprintf(stderr, "malloc failed!\n");
    exit(EXIT_FAILURE);
  }
  dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

  w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
  w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
  w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
  w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);

  w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
  w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
  w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

  w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

void read_checkpoint(char *checkpoint,
  Config *config,
  TransformerWeights *weights,
  int *fd,
  float **data,
  size_t *file_size)
{
  FILE *file = fopen(checkpoint, "rb");
  if (!file) {
    fprintf(stderr, "Couldn't open file %s\n", checkpoint);
    exit(EXIT_FAILURE);
  }
  // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
  uint32_t magic_number;
  if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { exit(EXIT_FAILURE); }
  if (magic_number != 0x616b3432) {
    fprintf(stderr, "Bad magic number\n");
    exit(EXIT_FAILURE);
  }
  // read in the version number (uint32), has to be 2
  int version;
  if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
  if (version != 2) {
    fprintf(stderr, "Bad version %d, need version 2\n", version);
    exit(EXIT_FAILURE);
  }
  int header_size = 256;// the header size for version 2 in bytes
  // read in the Config
  if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
  // read in flags
  uint8_t shared_classifier;// a byte to indicate if the classifier is shared
  if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { exit(EXIT_FAILURE); }
  int group_size;// the group size used in quantization
  if (fread(&group_size, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
  /* These constraints keep row groups aligned with the checkpoint layout. */
  int kv_dim = 0;
  if (config->dim <= 0 || config->hidden_dim <= 0 || config->n_layers <= 0 || config->n_heads <= 0
      || config->n_kv_heads <= 0 || config->vocab_size < 259 || config->seq_len <= 0 || config->dim % 4 != 0
      || group_size <= 0 || group_size > 65536 || config->dim % config->n_heads != 0
      || config->n_heads % config->n_kv_heads != 0 || (config->dim / config->n_heads) % 2 != 0
      || config->dim % group_size != 0 || config->hidden_dim % group_size != 0 || shared_classifier > 1) {
    fprintf(stderr, "Invalid configuration or group size in INT8 checkpoint\n");
    exit(EXIT_FAILURE);
  }
  kv_dim = (config->dim / config->n_heads) * config->n_kv_heads;
  /* Upstream tensor mapper and layer offsets use int indices. */
  if ((size_t)config->vocab_size * config->dim > INT_MAX
      || (size_t)config->dim * config->dim * config->n_layers > INT_MAX
      || (size_t)config->hidden_dim * config->dim * config->n_layers > INT_MAX
      || (size_t)config->seq_len * kv_dim * config->n_layers > INT_MAX
      || (size_t)config->n_heads * config->seq_len > INT_MAX) {
    fprintf(stderr, "Model too large for this small-model baseline\n");
    exit(EXIT_FAILURE);
  }
  GS = group_size;
  size_t quant_values = (size_t)config->vocab_size * config->dim * (shared_classifier ? 1 : 2);
  quant_values += (size_t)config->n_layers
                  * (2 * (size_t)config->dim * config->dim + 2 * (size_t)config->dim * kv_dim
                     + 3 * (size_t)config->hidden_dim * config->dim);
  size_t norm_values = (size_t)config->dim * (2 * (size_t)config->n_layers + 1);
  size_t expected_bytes = 256 + norm_values * sizeof(float) + quant_values + (quant_values / GS) * sizeof(float);
  // Verify the checkpoint size before loading its weight data.
  if (fseek(file, 0, SEEK_END) != 0) {
    fprintf(stderr, "Could not seek in checkpoint\n");
    fclose(file);
    exit(EXIT_FAILURE);
  }
  long length = ftell(file);
  if (length < 0 || (size_t)length != expected_bytes) {
    fprintf(stderr, "INT8 checkpoint size does not match its header\n");
    fclose(file);
    exit(EXIT_FAILURE);
  }
  *file_size = (size_t)length;
#ifdef _WIN32
  // Native MinGW has no POSIX mmap. Read once, before timed generation.
  *fd = -1;
  *data = malloc(*file_size);
  if (!*data) {
    fprintf(stderr, "Could not allocate checkpoint buffer\n");
    fclose(file);
    exit(EXIT_FAILURE);
  }
  if (fseek(file, 0, SEEK_SET) != 0 || fread(*data, 1, *file_size, file) != *file_size) {
    fprintf(stderr, "Could not read checkpoint\n");
    free(*data);
    fclose(file);
    exit(EXIT_FAILURE);
  }
  fclose(file);
#else
  fclose(file);
  *fd = open(checkpoint, O_RDONLY);
  if (*fd == -1) {
    fprintf(stderr, "open failed!\n");
    exit(EXIT_FAILURE);
  }
  *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
  if (*data == MAP_FAILED) {
    fprintf(stderr, "mmap failed!\n");
    exit(EXIT_FAILURE);
  }
#endif
  void *weights_ptr = ((char *)*data) + header_size;// skip header bytes. char is 1 byte
  memory_map_weights(weights, config, weights_ptr, shared_classifier);
}

void build_transformer(Transformer *t, char *checkpoint_path)
{
  // read in the Config and the Weights from the checkpoint
  read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
  // allocate the RunState buffers
  malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t)
{
  // free QuantizedTensors
  free(t->weights.q_tokens);
  free(t->weights.token_embedding_table);
  free(t->weights.wq);
  free(t->weights.wk);
  free(t->weights.wv);
  free(t->weights.wo);
  free(t->weights.w1);
  free(t->weights.w2);
  free(t->weights.w3);
  if (t->weights.wcls != t->weights.q_tokens) { free(t->weights.wcls); }
  // Release the checkpoint with the matching platform-specific operation.
#ifdef _WIN32
  free(t->data);
#else
  if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
  if (t->fd != -1) { close(t->fd); }
#endif
  // free the RunState buffers
  free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float *o, float *x, float *weight, int size)
{
  // calculate sum of squares
  float ss = 0.0f;
  for (int j = 0; j < size; j++) { ss += x[j] * x[j]; }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  // normalize and scale
  for (int j = 0; j < size; j++) { o[j] = weight[j] * (ss * x[j]); }
}

void softmax(float *x, int size)
{
  // find max value (for numerical stability)
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) { max_val = x[i]; }
  }
  // exp and sum
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) { x[i] /= sum; }
}

/* W has d rows and n columns. Scales are stored per ROW, then per group.
 * With n divisible by GS this matches the upstream version-2 file layout.
 */
void matmul(float *out, QuantizedTensor *x, QuantizedTensor *w, int n, int d)
{
  const int groups = (n + GS - 1) / GS;
  for (int row = 0; row < d; row++) {
    float result = 0.0f;
    for (int g = 0; g < groups; g++) {
      int32_t sum = 0;
      int start = g * GS;
      int count = n - start < GS ? n - start : GS;
      for (int k = 0; k < count; k++) {
        size_t wi = (size_t)row * n + start + k;
        sum += (int32_t)w->q[wi] * (int32_t)x->q[start + k];
      }
      /* Reset INT32 sum for each group; each group has different scales. */
      result += (float)sum * w->s[(size_t)row * groups + g] * x->s[g];
    }
    out[row] = result;
  }
}

float *forward(Transformer *transformer, int token, int pos)
{

  // a few convenience variables
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;
  float *x = s->x;
  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul = p->n_heads / p->n_kv_heads;// integer multiplier of the kv sharing in multiquery
  int hidden_dim = p->hidden_dim;
  int head_size = dim / p->n_heads;

  // copy the token embedding into x
  memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

  // forward all the layers
  for (int l = 0; l < p->n_layers; l++) {

    // attention rmsnorm
    rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

    // qkv matmuls for this position
    quantize(&s->xq, s->xb, dim);
    matmul(s->q, &s->xq, w->wq + l, dim, dim);
    matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
    matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    for (int i = 0; i < dim; i += 2) {
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      int rotn = i < kv_dim ? 2 : 1;// how many vectors? 2 = q & k, 1 = q only
      for (int v = 0; v < rotn; v++) {
        float *vec = v == 0 ? s->q : s->k;// the vector to rotate (query or key)
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }

    // save key,value at this time step (pos) to our kv cache
    int loff = l * p->seq_len * kv_dim;// kv cache layer offset for convenience
    float *key_cache_row = s->key_cache + loff + pos * kv_dim;
    float *value_cache_row = s->value_cache + loff + pos * kv_dim;
    memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
    memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

    // multihead attention. iterate over all heads
    int h;
    for (h = 0; h < p->n_heads; h++) {
      // get the query vector for this head
      float *q = s->q + h * head_size;
      // attention scores for this head
      float *att = s->att + h * p->seq_len;
      // iterate over all timesteps, including the current one
      for (int t = 0; t <= pos; t++) {
        // get the key vector for this head and at this timestep
        float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) { score += q[i] * k[i]; }
        score /= sqrtf(head_size);
        // save the score to the attention buffer
        att[t] = score;
      }

      // softmax the scores to get attention weights, from 0..pos inclusively
      softmax(att, pos + 1);

      // weighted sum of the values, store back into xb
      float *xb = s->xb + h * head_size;
      memset(xb, 0, head_size * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        // get the value vector for this head and at this timestep
        float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // get the attention weight for this timestep
        float a = att[t];
        // accumulate the weighted value into xb
        for (int i = 0; i < head_size; i++) { xb[i] += a * v[i]; }
      }
    }

    // final matmul to get the output of the attention
    quantize(&s->xq, s->xb, dim);
    matmul(s->xb2, &s->xq, w->wo + l, dim, dim);

    // residual connection back into x
    for (int i = 0; i < dim; i++) { x[i] += s->xb2[i]; }

    // ffn rmsnorm
    rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    quantize(&s->xq, s->xb, dim);
    matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
    matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

    // SwiGLU non-linearity
    for (int i = 0; i < hidden_dim; i++) {
      float val = s->hb[i];
      // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
      val *= (1.0f / (1.0f + expf(-val)));
      // elementwise multiply with w3(x)
      val *= s->hb2[i];
      s->hb[i] = val;
    }

    // final matmul to get the output of the ffn
    quantize(&s->hq, s->hb, hidden_dim);
    matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

    // residual connection
    for (int i = 0; i < dim; i++) { x[i] += s->xb[i]; }
  }

  // final rmsnorm
  rmsnorm(x, x, w->rms_final_weight, dim);

  // classifier into logits
  quantize(&s->xq, x, dim);
  matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
  return s->logits;
}

// Adapter to perplexity.h's forward_fn: perplexity.c only knows the model as a void*
static float *forward_cb(void *model, int token, int pos) { return forward((Transformer *)model, token, pos); }


// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct
{
  float prob;
  int index;
} ProbIndex;// struct used when sorting probabilities during top-p sampling

typedef struct
{
  int vocab_size;
  ProbIndex *probindex;// buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n)
{
  // return the index that has the highest probability
  int max_i = 0;
  float max_p = probabilities[0];
  for (int i = 1; i < n; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

int sample_mult(float *probabilities, int n, float coin)
{
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  float cdf = 0.0f;
  for (int i = 0; i < n; i++) {
    cdf += probabilities[i];
    if (coin < cdf) { return i; }
  }
  return n - 1;// in case of rounding errors
}

int compare(const void *a, const void *b)
{
  ProbIndex *a_ = (ProbIndex *)a;
  ProbIndex *b_ = (ProbIndex *)b;
  if (a_->prob > b_->prob) return -1;
  if (a_->prob < b_->prob) return 1;
  return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin)
{
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()

  int n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / (n - 1);
  for (int i = 0; i < n; i++) {
    if (probabilities[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = probabilities[i];
      n0++;
    }
  }
  qsort(probindex, n0, sizeof(ProbIndex), compare);

  // truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  int last_idx = n0 - 1;// in case of rounding errors consider all elements
  for (int i = 0; i < n0; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break;// we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) { return probindex[i].index; }
  }
  return probindex[last_idx].index;// in case of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed)
{
  sampler->vocab_size = vocab_size;
  sampler->temperature = temperature;
  sampler->topp = topp;
  sampler->rng_state = rng_seed;
  // buffer only used with nucleus sampling; may not need but it's ~small
  sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) { free(sampler->probindex); }

unsigned int random_u32(unsigned long long *state)
{
  // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
  *state ^= *state >> 12;
  *state ^= *state << 25;
  *state ^= *state >> 27;
  return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state)
{// random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits)
{
  // sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f) {
    // greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler->vocab_size);
  } else {
    // apply the temperature to the logits
    for (int q = 0; q < sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
    // apply softmax to the logits to get the probabilities for next token
    softmax(logits, sampler->vocab_size);
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // we sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      // simply sample from the predicted probability distribution
      next = sample_mult(logits, sampler->vocab_size, coin);
    } else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
    }
  }
  return next;
}

// ----------------------------------------------------------------------------
// utilities: time

double time_in_seconds(void)
{
#ifdef _WIN32
  LARGE_INTEGER ticks, frequency;
  QueryPerformanceCounter(&ticks);
  QueryPerformanceFrequency(&frequency);
  return (double)ticks.QuadPart / (double)frequency.QuadPart;
#else
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
#endif
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps)
{
  char *empty_prompt = "";
  if (prompt == NULL) { prompt = empty_prompt; }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int));// +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // start the main loop
  double start = 0.0;// used to time our code, only initialized after first iteration
  int next;// will store the next token in the sequence
  int token = prompt_tokens[0];// kick off with the first token in the prompt
  int pos = 0;// position in the sequence
  while (pos < steps) {

    // forward the transformer to get logits for the next token
    float *logits = forward(transformer, token, pos);

    // advance the state state machine
    if (pos < num_prompt_tokens - 1) {
      // if we are still processing the input prompt, force the next prompt token
      next = prompt_tokens[pos + 1];
    } else {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
    }
    pos++;

    // data-dependent terminating condition: the BOS (=1) token delimits sequences
    if (next == 1) { break; }

    // print the token as string, decode it with the Tokenizer object
    char *piece = decode(tokenizer, token, next);
    safe_printf(piece);// same as printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0.0) { start = time_in_seconds(); }
  }
  printf("\n");

  // report achieved tok/s (pos-1 because the timer starts after first iteration)
  // includes forward() (with runtime quantization), sampling and printing;
  // excludes checkpoint loading and the first step
  if (pos > 1) {
    double end = time_in_seconds();
    fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (end - start));
  }

  free(prompt_tokens);
}

void read_stdin(const char *guide, char *buffer, size_t bufsize)
{
  // read a line from stdin, up to but not including \n
  printf("%s", guide);
  if (fgets(buffer, bufsize, stdin) != NULL) {
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
      buffer[len - 1] = '\0';// strip newline
    }
  }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer,
  Tokenizer *tokenizer,
  Sampler *sampler,
  char *cli_user_prompt,
  char *cli_system_prompt,
  int steps)
{

  // buffers for reading the system prompt and user prompt from stdin
  // you'll notice they are soomewhat haphazardly and unsafely set atm
  char system_prompt[512];
  char user_prompt[512];
  char rendered_prompt[1152];
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc(1152 * sizeof(int));
  int user_idx;

  // start the main loop
  int8_t user_turn = 1;// user starts
  int next;// will store the next token in the sequence
  int token;// stores the current token to feed into the transformer
  int pos = 0;// position in the sequence
  while (pos < steps) {

    // when it is the user's turn to contribute tokens to the dialog...
    if (user_turn) {
      // get the (optional) system prompt at position 0
      if (pos == 0) {
        // at position 0, the user can also contribute a system prompt
        if (cli_system_prompt == NULL) {
          // system prompt was not passed in, attempt to get it from stdin
          read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
        } else {
          // system prompt was passed in, use it
          strcpy(system_prompt, cli_system_prompt);
        }
      }
      // get the user prompt
      if (pos == 0 && cli_user_prompt != NULL) {
        // user prompt for position 0 was passed in, use it
        strcpy(user_prompt, cli_user_prompt);
      } else {
        // otherwise get user prompt from stdin
        read_stdin("User: ", user_prompt, sizeof(user_prompt));
      }
      // render user/system prompts into the Llama 2 Chat schema
      if (pos == 0 && system_prompt[0] != '\0') {
        char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
        sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
      } else {
        char user_template[] = "[INST] %s [/INST]";
        sprintf(rendered_prompt, user_template, user_prompt);
      }
      // encode the rendered prompt into tokens
      encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
      user_idx = 0;// reset the user index
      user_turn = 0;
      printf("Assistant: ");
    }

    // determine the token to pass into the transformer next
    if (user_idx < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt token
      token = prompt_tokens[user_idx++];
    } else {
      // otherwise use the next token sampled from previous turn
      token = next;
    }
    // EOS (=2) token ends the Assistant turn
    if (token == 2) { user_turn = 1; }

    // forward the transformer to get logits for the next token
    float *logits = forward(transformer, token, pos);
    next = sample(sampler, logits);
    pos++;

    if (user_idx >= num_prompt_tokens && next != 2) {
      // the Assistant is responding, so print its output
      char *piece = decode(tokenizer, token, next);
      safe_printf(piece);// same as printf("%s", piece), but skips "unsafe" bytes
      fflush(stdout);
    }
    if (next == 2) { printf("\n"); }
  }
  printf("\n");
  free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage()
{
  fprintf(stderr, "Usage:   run_int8_baseline <checkpoint> [options]\n");
  fprintf(stderr, "Example: run_int8_baseline model_int8.bin -n 256 -i \"Once upon a time\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
  fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
  fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
  fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
  fprintf(stderr, "  -i <string> input prompt\n");
  fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
  fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
  fprintf(stderr, "  -f <string> text file to score in ppl mode\n");
  fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{

  // default parameters
  char *checkpoint_path = NULL;// e.g. out/model.bin
  char *tokenizer_path = "tokenizer.bin";
  float temperature = 1.0f;// 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f;// top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int steps = 256;// number of steps to run for
  char *prompt = NULL;// prompt string
  unsigned long long rng_seed = 0;// seed rng with time by default
  char *mode = "generate";// generate|chat
  char *system_prompt = NULL;// the (optional) system prompt to use in chat mode
  char *text_path = NULL;// evaluation text file for ppl mode

  // poor man's C argparse so we can override the defaults above from the command line
  if (argc >= 2) {
    checkpoint_path = argv[1];
  } else {
    error_usage();
  }
  for (int i = 2; i < argc; i += 2) {
    // do some basic validation
    if (i + 1 >= argc) { error_usage(); }// must have arg after flag
    if (argv[i][0] != '-') { error_usage(); }// must start with dash
    if (strlen(argv[i]) != 2) { error_usage(); }// must be -x (one dash, one letter)
    // read in the args
    if (argv[i][1] == 't') {
      temperature = atof(argv[i + 1]);
    } else if (argv[i][1] == 'p') {
      topp = atof(argv[i + 1]);
    } else if (argv[i][1] == 's') {
      rng_seed = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'n') {
      steps = atoi(argv[i + 1]);
    } else if (argv[i][1] == 'i') {
      prompt = argv[i + 1];
    } else if (argv[i][1] == 'z') {
      tokenizer_path = argv[i + 1];
    } else if (argv[i][1] == 'm') {
      mode = argv[i + 1];
    } else if (argv[i][1] == 'y') {
      system_prompt = argv[i + 1];
    } else if (argv[i][1] == 'f') {
      text_path = argv[i + 1];
    } else {
      error_usage();
    }
  }

  // parameter validation/overrides
  if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0) temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp) topp = 0.9;
  if (steps < 0) steps = 0;

  // build the Transformer via the model .bin file
  Transformer transformer;
  build_transformer(&transformer, checkpoint_path);
  if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;// override to ~max length

  // build the Tokenizer via the tokenizer .bin file
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

  // build the Sampler
  Sampler sampler;
  build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  // run!
  if (strcmp(mode, "generate") == 0) {
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
  } else if (strcmp(mode, "chat") == 0) {
    chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
  } else if (strcmp(mode, "ppl") == 0) {
    if (text_path == NULL) { error_usage(); }
    perplexity_file(
      &transformer, forward_cb, transformer.config.vocab_size, transformer.config.seq_len, &tokenizer, text_path);
  } else {
    fprintf(stderr, "unknown mode: %s\n", mode);
    error_usage();
  }

  // memory and file handles cleanup
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);
  return 0;
}
#endif
