#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#define CUDA_CHECK(call) do { cudaError_t error = call; if (error != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); cudaDeviceReset(); exit(EXIT_FAILURE); } } while(0)

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 8
#define LEARNING_RATE 0.01
#define NUM_TRAIN 60000
#define NUM_TEST  10000

typedef struct {
  float *weights1;      // First layer weights (INPUT_SIZE × HIDDEN_SIZE)
  float *weights2;      // Second layer weights (HIDDEN_SIZE × OUTPUT_SIZE)
  float *bias1;         // First layer bias (HIDDEN_SIZE)
  float *bias2;         // Second layer bias (OUTPUT_SIZE)
  float *grad_weights1; // First layer weight gradients
  float *grad_weights2; // Second layer weight gradients
  float *grad_bias1;    // First layer bias gradients
  float *grad_bias2;    // Second layer bias gradients
} NeuralNetwork;

// Allocate memory on GPU VRAM   
void initialize_neural_netowork(NeuralNetwork *nn) {
  CUDA_CHECK(cudaMalloc(&nn->weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&nn->weights1, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
  
  CUDA_CHECK(cudaMalloc(&nn->bias1, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&nn->bias2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&nn->grad_weight1, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&nn->grad_weight2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&nn->grad_bias1, OUTPUT_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&nn->grad_bias2, OUTPUT_SIZE * sizeof(float)));

  initalize_random_weights(nn);
}

void initalize_random_weights(NeuralNetwork *nn) {
  float *h_weights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
  init_weights(h_weights1, INPUT_SIZE * HIDDEN_SIZE);
  CUDA_CHECK(cudaMemcpy(nn->weight1s, h_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  free(h_weights1);

  float *h_weights1 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
  init_weights(h_weights1, HIDDEN_SIZE * OUTPUT_SIZE);
  CUDA_CHECK(cudaMemcpy(nn->weight1s, h_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  free(h_weights2);

  float *h_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
  initalize_bias(h_bias1, HIDDEN_SIZE);
  CUDA_CHECK(cudaMemcpy(nn->bias1, h_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  free(h_bias1);
  
  float *h_bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
  initalize_bias(h_bias1, OUTPUT_SIZE);
  CUDA_CHECK(cudaMemcpy(nn->bias2, h_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  free(h_bias2)
}

__global__ void bias_forward(float *x, float *bias, int batch_size, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int batch = idx / batch_size;
  int i = idx % size;

  if (batch < batch_size && i < size) {
    x[idx] += bias[i];
  }
}

__global__ void matmul_forward(float *A, float *B, float *C, int m, int n, int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < k) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
      sum += A[row*n+i] * B[i*k+col];
    }

    C[row * k + col] = sum;
  }


  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      C[i*k+j] = 0.0f;
      for (int l = 0; l < n; l++) { 
        C[i*k+j] += A[i*n + l] * B[l*k + j];
      }
    }
  }
}

// assume, (2x1) (2x3)
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

// assume, (1x2) (3x2)
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

void relu_backward(float *dY, float *activation, float *dX, int size) {
  for (int i = 0; i < size; i++) {
    dX[i] = activation[i] > 0.0f ? dY[i] : 0.0f;
  }
}

void bias_forward(float *Z, float *b, int m, int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      Z[i*k+j] += b[j];
    }
  }
}

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

float cross_entropy_loss(float *probs, int *y, int rows, int cols) {
  float total_loss = 0.0f; 
  for (int r = 0; r < rows; r++) {
    int target = y[r]; 
    total_loss += -logf(probs[r*cols+target]);
  }
  return total_loss / rows;
}

// params: weights or biases to update (flat)
// grads:  their gradients (same shape)
// SGD step: walk each parameter one small step opposite its gradient.
void sgd_update(float *params, float *grads, int size) {
  for (int i = 0; i < size; i++) {
    params[i] -= LEARNING_RATE * grads[i];
  }
}

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

// Uniform random in [-limit, limit]. Xavier-style: limit = sqrt(6/(fan_in+fan_out)).
void init_weights(float *W, int fan_in, int fan_out) {
  float limit = sqrtf(6.0f / (fan_in + fan_out));
  for (int i = 0; i < fan_in * fan_out; i++) {
    float u = (float)rand() / (float)RAND_MAX;   // [0,1]
    W[i] = (u * 2.0f - 1.0f) * limit;            // [-limit, limit]
  }
}

// The MNIST .bin files are raw dumps: X is float32 already normalized to [0,1]
void load_floats(const char *path, float *dst, int count) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); exit(1); }
  size_t got = fread(dst, sizeof(float), count, f);
  if ((int)got != count) { fprintf(stderr, "ERROR: %s short read %zu/%d\n", path, got, count); exit(1); }
  fclose(f);
}

void load_ints(const char *path, int *dst, int count) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); exit(1); }
  size_t got = fread(dst, sizeof(int), count, f);
  if ((int)got != count) { fprintf(stderr, "ERROR: %s short read %zu/%d\n", path, got, count); exit(1); }
  fclose(f);
}

int main(void) {
  srand(42); // seed 

  float *W1 = malloc(sizeof(float) * INPUT_SIZE  * HIDDEN_SIZE);
  float *b1 = calloc(HIDDEN_SIZE, sizeof(float));
  float *W2 = malloc(sizeof(float) * HIDDEN_SIZE * OUTPUT_SIZE);
  float *b2 = calloc(OUTPUT_SIZE, sizeof(float));
  init_weights(W1, INPUT_SIZE,  HIDDEN_SIZE);
  init_weights(W2, HIDDEN_SIZE, OUTPUT_SIZE);

  float *dW1 = malloc(sizeof(float) * INPUT_SIZE  * HIDDEN_SIZE);
  float *db1 = malloc(sizeof(float) * HIDDEN_SIZE);
  float *dW2 = malloc(sizeof(float) * HIDDEN_SIZE * OUTPUT_SIZE);
  float *db2 = malloc(sizeof(float) * OUTPUT_SIZE);

  float *z1 = malloc(sizeof(float) * BATCH_SIZE * HIDDEN_SIZE);
  float *z2 = malloc(sizeof(float) * BATCH_SIZE * OUTPUT_SIZE);

  float *dlogits = malloc(sizeof(float) * BATCH_SIZE * OUTPUT_SIZE);
  float *da1     = malloc(sizeof(float) * BATCH_SIZE * HIDDEN_SIZE);
  float *dz1     = malloc(sizeof(float) * BATCH_SIZE * HIDDEN_SIZE);
  float *dXin    = malloc(sizeof(float) * BATCH_SIZE * INPUT_SIZE);

  float *X = malloc(sizeof(float) * NUM_TRAIN * INPUT_SIZE);
  int   *y = malloc(sizeof(int)   * NUM_TRAIN);
  float *Xt = malloc(sizeof(float) * NUM_TEST * INPUT_SIZE);
  int   *yt = malloc(sizeof(int)   * NUM_TEST);
  
  load_floats("data/X_train.bin", X, NUM_TRAIN * INPUT_SIZE);
  load_ints  ("data/y_train.bin", y, NUM_TRAIN);
  load_floats("data/X_test.bin",  Xt, NUM_TEST * INPUT_SIZE);
  load_ints  ("data/y_test.bin",  yt, NUM_TEST);
  printf("loaded %d train / %d test samples\n", NUM_TRAIN, NUM_TEST);

  int num_batches = NUM_TRAIN / BATCH_SIZE;
  int epochs = 20;

  for (int epoch = 0; epoch < epochs; epoch++) {
    float epoch_loss = 0.0f;
    int correct = 0;

    for (int b = 0; b < num_batches; b++) {
      float *Xb = X + b * BATCH_SIZE * INPUT_SIZE;
      int   *yb = y + b * BATCH_SIZE;

      // forward
      matmul_forward(Xb, W1, z1, BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
      bias_forward(z1, b1, BATCH_SIZE, HIDDEN_SIZE);
      relu_forward(z1, BATCH_SIZE * HIDDEN_SIZE);           // z1 now = a1

      matmul_forward(z1, W2, z2, BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
      bias_forward(z2, b2, BATCH_SIZE, OUTPUT_SIZE);
      softmax(z2, BATCH_SIZE, OUTPUT_SIZE);                 // z2 now = probs

      epoch_loss += cross_entropy_loss(z2, yb, BATCH_SIZE, OUTPUT_SIZE);

      // accuracy: argmax of each row
      for (int r = 0; r < BATCH_SIZE; r++) {
        int best = 0;
        for (int c = 1; c < OUTPUT_SIZE; c++)
          if (z2[r * OUTPUT_SIZE + c] > z2[r * OUTPUT_SIZE + best]) best = c;
        if (best == yb[r]) correct++;
      }

      // backward 
      softmax_ce_grad(z2, yb, dlogits, BATCH_SIZE, OUTPUT_SIZE);

      // layer 2 consumes dlogits -> dW2, db2, da1
      linear_backward(z1, W2, dlogits, da1, dW2, db2, BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

      // relu: mask da1 by where a1 (=z1) was on
      relu_backward(da1, z1, dz1, BATCH_SIZE * HIDDEN_SIZE);

      // layer 1 consumes dz1 -> dW1, db1 (dXin unused)
      linear_backward(Xb, W1, dz1, dXin, dW1, db1, BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);

      // update 
      sgd_update(W1, dW1, INPUT_SIZE  * HIDDEN_SIZE);
      sgd_update(b1, db1, HIDDEN_SIZE);
      sgd_update(W2, dW2, HIDDEN_SIZE * OUTPUT_SIZE);
      sgd_update(b2, db2, OUTPUT_SIZE);
    }

    printf("epoch %2d | loss %.4f | acc %.1f%%\n", epoch, epoch_loss / num_batches, 100.0f * correct / (num_batches * BATCH_SIZE));
  }

  // test-set accuracy (forward only, no grads)
  int test_correct = 0;
  int test_batches = NUM_TEST / BATCH_SIZE;
  for (int b = 0; b < test_batches; b++) {
    float *Xb = Xt + b * BATCH_SIZE * INPUT_SIZE;
    int   *yb = yt + b * BATCH_SIZE;

    matmul_forward(Xb, W1, z1, BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);
    bias_forward(z1, b1, BATCH_SIZE, HIDDEN_SIZE);
    relu_forward(z1, BATCH_SIZE * HIDDEN_SIZE);
    matmul_forward(z1, W2, z2, BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    bias_forward(z2, b2, BATCH_SIZE, OUTPUT_SIZE);
    softmax(z2, BATCH_SIZE, OUTPUT_SIZE);

    for (int r = 0; r < BATCH_SIZE; r++) {
      int best = 0;
      for (int c = 1; c < OUTPUT_SIZE; c++)
        if (z2[r * OUTPUT_SIZE + c] > z2[r * OUTPUT_SIZE + best]) best = c;
      if (best == yb[r]) test_correct++;
    }
  }
  printf("---\ntest accuracy: %.2f%%\n", 100.0f * test_correct / (test_batches * BATCH_SIZE));

  free(W1); free(b1); free(W2); free(b2);
  free(dW1); free(db1); free(dW2); free(db2);
  free(z1); free(z2);
  free(dlogits); free(da1); free(dz1); free(dXin);
  free(X); free(y); free(Xt); free(yt);
  return 0;
}