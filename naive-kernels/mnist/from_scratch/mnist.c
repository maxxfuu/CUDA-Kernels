/* ==========================================================================
 *  MNIST 2-layer MLP  —  C FROM SCRATCH   (you write the math, with me)
 * ==========================================================================
 *
 *  Same network as before:  784 -> Linear -> 256 -> ReLU -> Linear -> 10
 *  In C there is no `@` operator. Every matmul is triple-nested loops. Every
 *  array is a flat 1-D block of floats, indexed by hand (row-major):
 *
 *      matrix M with `rows` rows and `cols` cols  ->  M[r*cols + c]
 *
 *  All tensors, flattened:
 *      X   (8,784)   -> X[b*784 + i]        W1 (784,256) -> W1[i*256 + h]
 *      z1  (8,256)   -> z1[b*256 + h]       b1 (256,)    -> b1[h]
 *      z2  (8,10)    -> z2[b*10 + o]        W2 (256,10)  -> W2[h*10 + o]
 *
 *  The harness at the bottom loads the SAME inputs/weights PyTorch used, runs
 *  your functions, and diffs the loss + gradients against the answer key in
 *  ref/. max|Δ| ~ 0  ==>  your math matches PyTorch.
 *
 *  Build & run:
 *      cd src/mnist/from_scratch
 *      gcc -O2 mnist.c -o mnist -lm && ./mnist
 * ========================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define B    8       /* batch size            */
#define IN   784     /* input features        */
#define HID  256     /* hidden units          */
#define OUT  10      /* classes               */
#define LR   0.01f   /* (used later for SGD)  */

#define REF "/Users/maxfu/Desktop/Code/CUDA-Kernels/src/mnist/from_scratch/ref/"

/* ---- tiny I/O + helpers (provided) ------------------------------------- */
static float *load_f32(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    float *p = (float *)malloc(sizeof(float) * n);
    if (fread(p, sizeof(float), n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f); return p;
}
static int *load_i32(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    int *p = (int *)malloc(sizeof(int) * n);
    if (fread(p, sizeof(int), n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f); return p;
}
/* max absolute difference between an array you computed and a ref .bin */
static float max_abs_diff(const float *got, const char *ref_path, int n) {
    float *want = load_f32(ref_path, n), m = 0.0f;
    for (int i = 0; i < n; i++) { float d = fabsf(got[i] - want[i]); if (d > m) m = d; }
    free(want); return m;
}
static void report(const char *name, float diff) {
    printf("  %s %-4s  max|Δ vs PyTorch| = %.3e\n", diff < 1e-4f ? "PASS" : "FAIL", name, diff);
}

/* ======================================================================== *
 *  FORWARD
 * ======================================================================== */

/* y = x @ W + b
 *   x    (rows, in)      x[r*in + i]
 *   W    (in, out)       W[i*out + o]
 *   b    (out,)          b[o]
 *   y    (rows, out)     y[r*out + o]  =  sum_i x[r*in+i]*W[i*out+o] + b[o]
 *
 * TODO: triple loop over (r, o, i). Initialize each y[r*out+o] to b[o], then
 *       accumulate the sum over i.
 */
void linear_forward(const float *x, const float *W, const float *b,
                    float *y, int rows, int in, int out) {
    /* TODO: your loops here */
    (void)x; (void)W; (void)b; (void)rows; (void)in; (void)out;
    memset(y, 0, sizeof(float) * rows * out);   /* placeholder */
}

/* WORKED EXAMPLE — this is the C loop idiom you'll copy for the others.
 * a = max(z, 0), elementwise over n entries. */
void relu(const float *z, float *a, int n) {
    for (int i = 0; i < n; i++)
        a[i] = z[i] > 0.0f ? z[i] : 0.0f;
}

/* probs = row-wise stable softmax of logits (rows, C).
 *   for each row r:  m = max over C; e_o = exp(logits - m); probs = e/sum(e)
 * TODO: implement the per-row max, exp, and normalize.
 */
void softmax(const float *logits, float *probs, int rows, int C) {
    /* TODO */
    (void)logits; (void)rows; (void)C;
    memset(probs, 0, sizeof(float) * rows * C);  /* placeholder */
}

/* mean cross-entropy:  -1/B * sum_r log(probs[r, y[r]])
 * TODO: sum the log-prob of the correct class per row, negate, divide by rows.
 */
float cross_entropy(const float *probs, const int *y, int rows, int C) {
    /* TODO */
    (void)probs; (void)y; (void)rows; (void)C;
    return 0.0f;   /* placeholder */
}

/* ======================================================================== *
 *  BACKWARD
 * ======================================================================== */

/* dlogits = (probs - onehot(y)) / B         shape (rows, C)
 * TODO: copy probs into dlogits, subtract 1 at the correct class of each row,
 *       then divide everything by rows.
 */
void softmax_ce_grad(const float *probs, const int *y, float *dlogits, int rows, int C) {
    /* TODO */
    (void)probs; (void)y; (void)rows; (void)C;
    memset(dlogits, 0, sizeof(float) * rows * C);  /* placeholder */
}

/* WORKED EXAMPLE — relu backward: grad_in = grad_out * (z > 0). */
void relu_backward(const float *grad_out, const float *z, float *grad_in, int n) {
    for (int i = 0; i < n; i++)
        grad_in[i] = z[i] > 0.0f ? grad_out[i] : 0.0f;
}

/* Backward through y = x@W + b.  Given grad_out = dL/dy, produce:
 *   dW[i*out+o] = sum_r x[r*in+i] * grad_out[r*out+o]     (in, out)
 *   db[o]       = sum_r grad_out[r*out+o]                 (out,)
 *   dx[r*in+i]  = sum_o grad_out[r*out+o] * W[i*out+o]    (rows, in)
 * TODO: three loop nests (or fuse them). Zero dW/db/dx first, then accumulate.
 */
void linear_backward(const float *grad_out, const float *x, const float *W,
                     float *dx, float *dW, float *db, int rows, int in, int out) {
    /* TODO */
    (void)grad_out; (void)x; (void)W; (void)rows; (void)in; (void)out;
    memset(dx, 0, sizeof(float) * rows * in);
    memset(dW, 0, sizeof(float) * in * out);
    memset(db, 0, sizeof(float) * out);
}

/* ======================================================================== *
 *  HARNESS (provided) — loads answer key, runs your code, grades it
 * ======================================================================== */
int main(void) {
    /* inputs + params (same state PyTorch used) */
    float *X  = load_f32(REF "X.bin",  B * IN);
    int   *y  = load_i32(REF "y.bin",  B);
    float *W1 = load_f32(REF "W1.bin", IN * HID);
    float *b1 = load_f32(REF "b1.bin", HID);
    float *W2 = load_f32(REF "W2.bin", HID * OUT);
    float *b2 = load_f32(REF "b2.bin", OUT);

    /* scratch */
    float *z1 = malloc(sizeof(float) * B * HID);
    float *a1 = malloc(sizeof(float) * B * HID);
    float *z2 = malloc(sizeof(float) * B * OUT);
    float *pr = malloc(sizeof(float) * B * OUT);

    /* ---- forward ---- */
    linear_forward(X, W1, b1, z1, B, IN, HID);
    relu(z1, a1, B * HID);
    linear_forward(a1, W2, b2, z2, B, HID, OUT);
    softmax(z2, pr, B, OUT);
    float loss = cross_entropy(pr, y, B, OUT);

    float want_loss; { FILE *f = fopen(REF "loss.bin", "rb"); fread(&want_loss, 4, 1, f); fclose(f); }
    printf("\nFORWARD\n");
    printf("  loss = %.6f   (want %.6f,  |Δ| = %.3e)\n",
           loss, want_loss, fabsf(loss - want_loss));

    /* ---- backward ---- */
    float *dz2 = malloc(sizeof(float) * B * OUT);
    float *da1 = malloc(sizeof(float) * B * HID);
    float *dz1 = malloc(sizeof(float) * B * HID);
    float *dX  = malloc(sizeof(float) * B * IN);
    float *dW1 = malloc(sizeof(float) * IN * HID);
    float *db1 = malloc(sizeof(float) * HID);
    float *dW2 = malloc(sizeof(float) * HID * OUT);
    float *db2 = malloc(sizeof(float) * OUT);

    softmax_ce_grad(pr, y, dz2, B, OUT);
    linear_backward(dz2, a1, W2, da1, dW2, db2, B, HID, OUT);
    relu_backward(da1, z1, dz1, B * HID);
    linear_backward(dz1, X, W1, dX, dW1, db1, B, IN, HID);

    printf("BACKWARD\n");
    report("dW1", max_abs_diff(dW1, REF "dW1.bin", IN * HID));
    report("db1", max_abs_diff(db1, REF "db1.bin", HID));
    report("dW2", max_abs_diff(dW2, REF "dW2.bin", HID * OUT));
    report("db2", max_abs_diff(db2, REF "db2.bin", OUT));
    printf("\n");
    return 0;
}
