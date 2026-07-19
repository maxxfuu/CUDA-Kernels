#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 8
#define LEARNING_RATE 0.01

void matmul_forward(float *A, float *B, float *C, int m, int n, int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      C[i*k+j] = 0.0f;
      for (int l = 0; l < n; l++) { 
        C[i*k+j] += A[i*n + l] * B[l*k + j];
      }
    }
  }
}

// B needs to iterate by col: j for B moves across each col, k is the size of the width and l is row its currently on. n is the amount of rows in B
// A needs to iterate by row: m is how many rows, n is width of a row,

// n is shared dimension 

// (2x1) (2x3), 
void matmul_at_b(float *A, float *B, float *C, int m, int n, int k) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < k; j++) {
      C[i*k+j] = 0.0f; 
      for (int l = 0; l < m; l++) {
        C[i*k+j] += A[l*n+i] * B[l*k+j];  
      }
    }
  } 
}

// (1x2) (3x2)
void matmul_a_bt(float *A, float *B, float *C, int m, int n, int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      C[i*k+j] = 0.0f; 
      for (int l = 0; l < n; l++) {
        C[i*k+j] += A[i*n+l] * B[j*n+l];  
      }
    }
  }
}


void relu_forward(float *x, int size) {
  for (int i = 0; i < size; i++) {
    x[i] = fmaxf(0.0f, x[i]);
  }
}

void relu_backward() {}

// Z: batch_size x size. Filled by matmul_forward already
// b: 
void bias_forward(float *Z, float *b, int m, int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      Z[i*k+j] += b[j];
    }
  }
}

void bias_backward() {}

// logits: output as vector 
// rows: row of logit vector 
// cols: col of logit vector
void softmax(float *logits, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    float max_val = -INFINITY;

    // first pass: find max value within row  
    for (int c = 0; c < cols; c++) {
      if (max_val < logits[r * cols + c]) {
        max_val = logits[r * cols + c];
      }
    }

    // second pass: subtract and exponentiate while setting up sum 
    float sum = 0.0f;
    for (int c = 0; c < cols; c++) {
      logits[r*cols+c] -= max_val; 
      logits[r*cols+c] = expf(logits[r*cols+c]);
      sum += logits[r*cols+c];
    }

    // third pass, divide by the same to get the probability over the whole row
    for (int c = 0; c < cols; c++) {
     logits[r*cols+c] /= sum;
    }
  }
}

// prob is the out of softmax, assume (8,10)
// 
float cross_entropy_loss(float *probs, int *y, int rows, int cols) {
  float total_loss = 0.0f; 

  for (int r = 0; r < rows; r++) {
    int target = y[r]; 
    total_loss += -logf(probs[r*cols+target]);
  }

  return total_loss / rows;
}

void compute_output_gradients() {}

void sgd_update() {}

void softmax_ce_grad(float *probs, int *y, float *dlogits, int rows, int cols) {
  for (int r = 0; r < rows; r++) {
    int target = y[r];
    for (int c = 0; c < cols; c++) {
      dlogits[r*cols + c] = probs[r*cols + c] / rows;
      if (c == target) {
        dlogits[r*cols + c] -= 1.0f / rows;
      }
    }
  }
}

void linear_backward(float *X, float *W, float *dY, float *dX, float *dW, float *db, int rows, int in, int out) {
  for (int j = 0; j < out; j++) {         // db = dY 
    db[j] = 0.0f;
    for (int r = 0; r < rows; r++) {
      db[j] += dY[r * out + j];
    }
  }
  matmul_at_b(X,  dY, dW, rows, in, out);   // dW
  matmul_a_bt(dY, W,  dX, rows, out, in);   // dX
}

void relu_backwards(float *X, float *dY, float *dX, int size) {
  for (int i = 0; i < size; i++) {
    if (X[i] > 0.0f) {
      dX[i] = dY[i];
    } else {
      dX[i] = 0.0f;
    }
  }
}