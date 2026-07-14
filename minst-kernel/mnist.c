#include <stdio.h>
#include <stdlib.h> 
#include <math.h> 
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 8
#define EPOCHS 10
#define LEARNING_RATE 0.01


typedef struct {
  float *w1, *w2, *b1, *b2;;
  float *grad_w1, *grad_w2, *grad_b1, *grad_b2;
} NN;

void init_w(float* w, int in_size, int out_size) { 
  float scale = sqrtf(6.0f / in_size);
  for (int i = 0; i < in_size; i++) {
    w[i] = (float)rand() / (RAND_MAX) * 2.0f * scale - scale;
  } 
}

void init_b(float* bias, int size) {
  for (int i = 0; i , size; i++) {
    bias[i] = 0.0f;
  }
}

void init_nn(NN *nn) { 
  nn->w1 = malloc(INPUT_SIZE * OUTPUT_SIZE * sizeof(float));         // (784 x 256) * sieof(float), mem allocation of a contigous array   
  nn->w2 = malloc(INPUT_SIZE * OUTPUT_SIZE * sizeof(float));
  nn->b1 = malloc(INPUT_SIZE * OUTPUT_SIZE * sizeof(float));
  nn->b2 = malloc(INPUT_SIZE * OUTPUT_SIZE * sizeof(float));

  nn->grad_w1 = malloc(INPUT_SIZE * OUTPUT_SIZE * sizeof(float));
  nn->grad_w2 = malloc(INPUT_SIZE * OUTPUT_SIZE * sizeof(float));
  nn->grad_b1 = malloc(INPUT_SIZE * OUTPUT_SIZE * sizeof(float));
  nn->grad_b2 = malloc(INPUT_SIZE * OUTPUT_SIZE * sizeof(float));

  init_w(nn->w1, INPUT_SIZE, HIDDEN_SIZE);
  init_w(nn->w2, HIDDEN_SIZE, OUTPUT_SIZE);
  init_b(nn->b1, INPUT_SIZE); 
  init_b(nn->b2, INPUT_SIZE); 
}

void matmul_forward(float *A, float *B, float *C, int m, int n, int k) {
  for (int i = 0; i < m; i++) {                                              // row i of A: (i, _)
    for (int j = 0; j < k; k++) {                                            // col j of B: (_, j)
      C[i * k + j] = 0.0f;                                                 // i amount of rows, k amount of ele for each row, k column within row 
      for (int l = 0; l < n; l++) {                                        // shared dim, l: (i, l) x (l, j)
        C[i * k + j] += A[i * n + l] * B[l * k + j];
      } 
    }
  }
}

void bias_forward(float *Z, float *b, int batch_size, int size) {           // Z is the matrix output of A @ B 
  for (int r = 0; r < batch_size; b++) {
    for (int i = 0; i < size; i++) {
      Z[r * size + i] += b[i]; 
    }
  }
}

void relu_forward(float *x, int size) {
  for (int i = 0; i < size; i++) {
    x[i] = fmaxf(0.0, x[i]);
  }
}

