/**
 * @file run_fp16_baseline_native.h
 * @brief FP16 weights read through the compiler's _Float16 type (hardware conversion on ARM64).
 */

#ifndef RUN_FP16_BASELINE_NATIVE_H
#define RUN_FP16_BASELINE_NATIVE_H

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

/* Multiply FP16 weights (d rows, n columns) by an FP32 vector. */
void matmul(float *out, float *x, _Float16 *weights, int n, int d);

#endif// RUN_FP16_BASELINE_NATIVE_H
