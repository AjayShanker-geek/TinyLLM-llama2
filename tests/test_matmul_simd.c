/**
 * @file test_matmul_simd.c
 * @brief Kernel correctness test for the NEON FP32 matmul kernels.
 *
 * Compares matmul_basic and matmul_4row against a naive scalar matmul
 * (the same loop as llama2.c's run.c) on random data, for the model's
 * real matrix shapes and for odd shapes that exercise the leftover
 * row/column loops.
 *
 * Build and run from the project root (AddressSanitizer catches any read
 * or write past the end of W, x or xout):
 *   clang -O0 -g -fsanitize=address -Iinclude \
 *       -o build/test_matmul_simd tests/test_matmul_simd.c -lm
 *   ./build/test_matmul_simd
 **/

// Pull in the SIMD kernels without the program's main()
#define TESTING
#include "../src/run_fp32_simd.c"

// ----------------------------------------------------------------------------
// reference kernel: plain scalar loop, identical to run.c's matmul

static void matmul_naive(float* xout, float* x, float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[(size_t)i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// ----------------------------------------------------------------------------
// helpers

// random float in [-1, 1]
static float rand_unit(void) {
    return 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
}

// Compare one kernel's output against the reference. Returns 1 if every row
// is within tolerance. SIMD adds the products in a different order, so small
// rounding differences are expected; they grow with the row length n.
static int check_output(const char* name, float* got, float* ref, int n, int d) {
    const float tol = 1e-5f * (float)n;
    float max_diff = 0.0f;
    int bad_row = -1;
    for (int i = 0; i < d; i++) {
        float diff = fabsf(got[i] - ref[i]);
        // written as !(diff <= tol) so a NaN (an output row never written) also fails
        if (!(diff <= tol)) {
            bad_row = i;
            max_diff = diff;
            break;
        }
        if (diff > max_diff) max_diff = diff;
    }
    if (bad_row >= 0) {
        printf("  FAIL  %-12s n=%-5d d=%-5d row %d: got %g, expected %g\n",
               name, n, d, bad_row, got[bad_row], ref[bad_row]);
        return 0;
    }
    printf("  PASS  %-12s n=%-5d d=%-5d max diff %.3g (tol %.3g)\n", name, n, d, max_diff, tol);
    return 1;
}

// Run all kernels on one (n, d) shape. Returns the number of failures.
static int test_shape(int n, int d) {
    // Exact-size allocations, so AddressSanitizer flags any out-of-bounds access
    float* w = malloc((size_t)n * d * sizeof(float));
    float* x = malloc((size_t)n * sizeof(float));
    float* ref = malloc((size_t)d * sizeof(float));
    float* out = malloc((size_t)d * sizeof(float));
    if (!w || !x || !ref || !out) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }

    for (size_t k = 0; k < (size_t)n * d; k++) w[k] = rand_unit();
    for (int j = 0; j < n; j++) x[j] = rand_unit();

    matmul_naive(ref, x, w, n, d);

    int failures = 0;

    // Fill the output with NaN first so a row the kernel skips is caught
    for (int i = 0; i < d; i++) out[i] = NAN;
    matmul_basic(out, x, w, n, d);
    failures += !check_output("matmul_basic", out, ref, n, d);

    for (int i = 0; i < d; i++) out[i] = NAN;
    matmul_4row(out, x, w, n, d);
    failures += !check_output("matmul_4row", out, ref, n, d);

    free(w);
    free(x);
    free(ref);
    free(out);
    return failures;
}

// ----------------------------------------------------------------------------

int main(void) {
    srand(42); // fixed seed: every run tests the same numbers

    // {n (columns), d (rows)}
    const int shapes[][2] = {
        // tiny: SIMD loop runs zero or one time
        {1, 1}, {3, 2}, {4, 4}, {4, 1},
        // every leftover-column case: n % 4 = 1, 2, 3
        {5, 8}, {6, 8}, {7, 8}, {769, 8},
        // every leftover-row case for matmul_4row: d % 4 = 1, 2, 3
        {8, 1}, {8, 2}, {8, 3}, {8, 5}, {8, 13},
        // leftover rows and columns together
        {7, 7}, {769, 13},
        // the shapes stories110M actually uses
        {768, 768},    // wq, wk, wv, wo
        {768, 2048},   // w1, w3
        {2048, 768},   // w2
        {768, 32000},  // classifier
    };
    const int n_shapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

    int failures = 0;
    for (int s = 0; s < n_shapes; s++) {
        failures += test_shape(shapes[s][0], shapes[s][1]);
    }

    if (failures == 0) {
        printf("All %d shapes passed.\n", n_shapes);
        return 0;
    }
    printf("%d check(s) FAILED.\n", failures);
    return 1;
}
