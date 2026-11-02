/**
 * @file run_fp16_baseline.h
 */

#ifndef RUN_FP16_BASELINE_H
#define RUN_FP16_BASELINE_H

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

/* Decode one IEEE binary16 bit pattern into an FP32 value. */
float fp16_to_fp32(uint16_t bits);

/* Multiply FP16 weights (d rows, n columns) by an FP32 vector. */
void matmul(float *out, float *x, uint16_t *weights, int n, int d);

#endif// RUN_FP16_BASELINE_H
