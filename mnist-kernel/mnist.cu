#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error), error); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

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

void initalize_random_weights(NeuralNetwork *nn);
void initialize_weights(float *weights, int input_size, int output_size);

// Allocate memory on GPU VRAM
void initialize_neural_network(NeuralNetwork *nn) {
  CUDA_CHECK(cudaMalloc(&nn->weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&nn->weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
  
  CUDA_CHECK(cudaMalloc(&nn->bias1, HIDDEN_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&nn->bias2, OUTPUT_SIZE * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&nn->grad_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&nn->grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&nn->grad_bias1, HIDDEN_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&nn->grad_bias2, OUTPUT_SIZE * sizeof(float)));

  initalize_random_weights(nn);
}

void initalize_random_weights(NeuralNetwork *nn) {
  float *h_weights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
  initialize_weights(h_weights1, INPUT_SIZE, HIDDEN_SIZE);
  CUDA_CHECK(cudaMemcpy(nn->weights1, h_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  free(h_weights1);

  float *h_weights2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
  initialize_weights(h_weights2, HIDDEN_SIZE, OUTPUT_SIZE);
  CUDA_CHECK(cudaMemcpy(nn->weights2, h_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  free(h_weights2);

  float *h_bias1 = (float *)calloc(HIDDEN_SIZE, sizeof(float));
  CUDA_CHECK(cudaMemcpy(nn->bias1, h_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  free(h_bias1);
  
  float *h_bias2 = (float *)calloc(OUTPUT_SIZE, sizeof(float));
  CUDA_CHECK(cudaMemcpy(nn->bias2, h_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  free(h_bias2);
}

__global__ void bias_forward(float *x, float *bias, int batch_size, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int batch = idx / size;
  int i = idx % size;

  if (batch < batch_size && i < size) {
    x[idx] += bias[i];
  }
}

__global__ void matmul_forward(float *A, float *B, float *C, int m, int n, int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < k) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
      sum += A[row*n+i] * B[i*k+col];
    }
    C[row * k + col] = sum;
  }
}

__global__ void matmul_at_b(float *A, float *B, float *C, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < k) {
    float sum = 0.0f;
    for (int i = 0; i < m; i++) {
      sum += A[i*n+row] * B[i*k+col];  
    }
    C[row*k+col] = sum;
  }
}

__global__ void matmul_a_bt(float *A, float *B, float *C, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < k) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
      sum += A[row*n+i] * B[col*n+i];  
    }
    C[row*k+col] = sum;
  }
}

__global__ void relu_forward(float *x, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < rows && col < cols) {
      int idx = row * cols + col;
      x[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void relu_backward(float *dY, float *activation, float *dX, int rows, int cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x; 

  if (row < rows && col < cols) {
    int idx = row * cols + col;
    dX[idx] = activation[idx] > 0.0f ? dY[idx] : 0.0f;
  }
}

__global__ void bias_backward(float *db, float *dY, int rows, int out) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < out) {
    float sum = 0.0f;

    for (int row = 0; row < rows; row++) {
      sum += dY[row * out + col];
    }
    db[col] = sum;
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

__global__ void sgd_update(float *params, float *grads, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    params[idx] -= LEARNING_RATE * grads[idx];
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
  dim3 block(16, 16);

  dim3 grid_dw((out + block.x - 1) / block.x, (in + block.y - 1) / block.y);
  matmul_at_b<<<grid_dw, block>>>(X, dY, dW, rows, in, out);

  dim3 grid_dx((in + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  matmul_a_bt<<<grid_dx, block>>>(dY, W, dX, rows, out, in);

  int threads = 256;
  int blocks = (out + threads - 1) / threads;

  bias_backward<<<blocks, threads>>>(db, dY, rows, out);
  CUDA_CHECK(cudaGetLastError());
}

void initialize_weights(float *weights, int input_size, int output_size) {
  float scale = sqrtf(6.0f / input_size);
  for (int i = 0; i < input_size * output_size; i++) {
    weights[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
  }
}

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
  NeuralNetwork nn; 
  initialize_neural_network(&nn);

  // initialize cuBLAS context
  cublasHandle_t cublas_handle; 
  cublasCreate(&cublas_handle);

  float *z1, *z2, *dlogits, *da1, *dz1, *dXin;
  CUDA_CHECK(cudaMalloc(&z1, sizeof(float) * BATCH_SIZE * HIDDEN_SIZE));
  CUDA_CHECK(cudaMalloc(&z2, sizeof(float) * BATCH_SIZE * OUTPUT_SIZE));
  CUDA_CHECK(cudaMalloc(&dlogits, sizeof(float) * BATCH_SIZE * OUTPUT_SIZE));
  CUDA_CHECK(cudaMalloc(&da1, sizeof(float) * BATCH_SIZE * HIDDEN_SIZE));
  CUDA_CHECK(cudaMalloc(&dz1, sizeof(float) * BATCH_SIZE * HIDDEN_SIZE));
  CUDA_CHECK(cudaMalloc(&dXin, sizeof(float) * BATCH_SIZE * INPUT_SIZE));

  float *h_X = (float *)malloc(sizeof(float) * NUM_TRAIN * INPUT_SIZE);
  load_floats("data/X_train.bin", h_X, NUM_TRAIN * INPUT_SIZE);
  float *d_X;
  CUDA_CHECK(cudaMalloc(&d_X, sizeof(float) * NUM_TRAIN * INPUT_SIZE));
  CUDA_CHECK(cudaMemcpy(d_X, h_X, sizeof(float) * NUM_TRAIN * INPUT_SIZE, cudaMemcpyHostToDevice));
  free(h_X);

  int *h_y = (int *)malloc(sizeof(int) * NUM_TRAIN);
  load_ints("data/y_train.bin", h_y, NUM_TRAIN);
  int *d_y;
  CUDA_CHECK(cudaMalloc(&d_y, sizeof(int) * NUM_TRAIN));
  CUDA_CHECK(cudaMemcpy(d_y, h_y, sizeof(int) * NUM_TRAIN, cudaMemcpyHostToDevice));
  free(h_y);

  float *h_Xt = (float *)malloc(sizeof(float) * NUM_TEST * INPUT_SIZE);
  load_floats("data/X_test.bin", h_Xt, NUM_TEST * INPUT_SIZE);
  float *d_Xt;
  CUDA_CHECK(cudaMalloc(&d_Xt, sizeof(float) * NUM_TEST * INPUT_SIZE));
  CUDA_CHECK(cudaMemcpy(d_Xt, h_Xt, sizeof(float) * NUM_TEST * INPUT_SIZE, cudaMemcpyHostToDevice));
  free(h_Xt);

  int *h_yt = (int *)malloc(sizeof(int) * NUM_TEST);
  load_ints("data/y_test.bin", h_yt, NUM_TEST);
  int *d_yt;
  CUDA_CHECK(cudaMalloc(&d_yt, sizeof(int) * NUM_TEST));
  CUDA_CHECK(cudaMemcpy(d_yt, h_yt, sizeof(int) * NUM_TEST, cudaMemcpyHostToDevice));
  free(h_yt);

  printf("loaded %d train / %d test samples\n", NUM_TRAIN, NUM_TEST);

  float *h_z2      = (float *)malloc(sizeof(float) * BATCH_SIZE * OUTPUT_SIZE);
  float *h_dlogits = (float *)malloc(sizeof(float) * BATCH_SIZE * OUTPUT_SIZE);
  int   *h_yb      = (int *)malloc(sizeof(int)   * BATCH_SIZE);

  int num_batches = NUM_TRAIN / BATCH_SIZE;
  int epochs = 20;

  for (int epoch = 0; epoch < epochs; epoch++) {
    float epoch_loss = 0.0f;
    int correct = 0;

    for (int b = 0; b < num_batches; b++) {
      float *Xb = d_X + b * BATCH_SIZE * INPUT_SIZE;
      int   *yb = d_y + b * BATCH_SIZE;

      // forward
      dim3 block2d(16, 16);
      dim3 grid_l1((HIDDEN_SIZE + block2d.x - 1) / block2d.x, (BATCH_SIZE + block2d.y - 1) / block2d.y);
      matmul_forward<<<grid_l1, block2d>>>(Xb, nn.weights1, z1, BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);

      int threads1d = 256;
      int bias1_blocks = (BATCH_SIZE * HIDDEN_SIZE + threads1d - 1) / threads1d;
      bias_forward<<<bias1_blocks, threads1d>>>(z1, nn.bias1, BATCH_SIZE, HIDDEN_SIZE);

      relu_forward<<<grid_l1, block2d>>>(z1, BATCH_SIZE, HIDDEN_SIZE);           // z1 now = a1

      dim3 grid_l2((OUTPUT_SIZE + block2d.x - 1) / block2d.x, (BATCH_SIZE + block2d.y - 1) / block2d.y);
      matmul_forward<<<grid_l2, block2d>>>(z1, nn.weights2, z2, BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

      int bias2_blocks = (BATCH_SIZE * OUTPUT_SIZE + threads1d - 1) / threads1d;
      bias_forward<<<bias2_blocks, threads1d>>>(z2, nn.bias2, BATCH_SIZE, OUTPUT_SIZE);
      CUDA_CHECK(cudaGetLastError());

      // pull this batch's logits + labels to host — softmax/loss/argmax/grad
      // below are plain CPU functions, they can't read device pointers.
      CUDA_CHECK(cudaMemcpy(h_z2, z2, sizeof(float) * BATCH_SIZE * OUTPUT_SIZE, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_yb, yb, sizeof(int) * BATCH_SIZE, cudaMemcpyDeviceToHost));

      softmax(h_z2, BATCH_SIZE, OUTPUT_SIZE);                 // h_z2 now = probs

      epoch_loss += cross_entropy_loss(h_z2, h_yb, BATCH_SIZE, OUTPUT_SIZE);

      // accuracy: argmax of each row
      for (int r = 0; r < BATCH_SIZE; r++) {
        int best = 0;
        for (int c = 1; c < OUTPUT_SIZE; c++)
          if (h_z2[r * OUTPUT_SIZE + c] > h_z2[r * OUTPUT_SIZE + best]) best = c;
        if (best == h_yb[r]) correct++;
      }

      // backward, compute dlogits on host, then push it back to device
      // for linear_backward, which expects a device pointer.
      softmax_ce_grad(h_z2, h_yb, h_dlogits, BATCH_SIZE, OUTPUT_SIZE);
      CUDA_CHECK(cudaMemcpy(dlogits, h_dlogits, sizeof(float) * BATCH_SIZE * OUTPUT_SIZE, cudaMemcpyHostToDevice));

      // layer 2 consumes dlogits -> nn.grad_weights2, nn.grad_bias2, da1
      linear_backward(z1, nn.weights2, dlogits, da1, nn.grad_weights2, nn.grad_bias2, BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

      // relu: mask da1 by where a1 (=z1) was on
      relu_backward<<<grid_l1, block2d>>>(da1, z1, dz1, BATCH_SIZE, HIDDEN_SIZE);

      // layer 1 consumes dz1 -> nn.grad_weights1, nn.grad_bias1 (dXin unused)
      linear_backward(Xb, nn.weights1, dz1, dXin, nn.grad_weights1, nn.grad_bias1, BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);

      // update
      int w1_size = INPUT_SIZE * HIDDEN_SIZE;
      sgd_update<<<(w1_size + threads1d - 1) / threads1d, threads1d>>>(nn.weights1, nn.grad_weights1, w1_size);
      sgd_update<<<(HIDDEN_SIZE + threads1d - 1) / threads1d, threads1d>>>(nn.bias1, nn.grad_bias1, HIDDEN_SIZE);

      int w2_size = HIDDEN_SIZE * OUTPUT_SIZE;
      sgd_update<<<(w2_size + threads1d - 1) / threads1d, threads1d>>>(nn.weights2, nn.grad_weights2, w2_size);
      sgd_update<<<(OUTPUT_SIZE + threads1d - 1) / threads1d, threads1d>>>(nn.bias2, nn.grad_bias2, OUTPUT_SIZE);
      CUDA_CHECK(cudaGetLastError());
    }

    printf("epoch %2d | loss %.4f | acc %.1f%%\n", epoch, epoch_loss / num_batches, 100.0f * correct / (num_batches * BATCH_SIZE));
  }

  // test-set accuracy (forward only, no grads)
  int test_correct = 0;
  int test_batches = NUM_TEST / BATCH_SIZE;
  for (int b = 0; b < test_batches; b++) {
    float *Xb = d_Xt + b * BATCH_SIZE * INPUT_SIZE;
    int   *yb = d_yt + b * BATCH_SIZE;

    dim3 block2d(16, 16);
    dim3 grid_l1((HIDDEN_SIZE + block2d.x - 1) / block2d.x, (BATCH_SIZE + block2d.y - 1) / block2d.y);
    matmul_forward<<<grid_l1, block2d>>>(Xb, nn.weights1, z1, BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE);

    int threads1d = 256;
    bias_forward<<<(BATCH_SIZE * HIDDEN_SIZE + threads1d - 1) / threads1d, threads1d>>>(z1, nn.bias1, BATCH_SIZE, HIDDEN_SIZE);

    relu_forward<<<grid_l1, block2d>>>(z1, BATCH_SIZE, HIDDEN_SIZE);

    dim3 grid_l2((OUTPUT_SIZE + block2d.x - 1) / block2d.x, (BATCH_SIZE + block2d.y - 1) / block2d.y);
    matmul_forward<<<grid_l2, block2d>>>(z1, nn.weights2, z2, BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    bias_forward<<<(BATCH_SIZE * OUTPUT_SIZE + threads1d - 1) / threads1d, threads1d>>>(z2, nn.bias2, BATCH_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_z2, z2, sizeof(float) * BATCH_SIZE * OUTPUT_SIZE, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_yb, yb, sizeof(int) * BATCH_SIZE, cudaMemcpyDeviceToHost));

    softmax(h_z2, BATCH_SIZE, OUTPUT_SIZE);

    for (int r = 0; r < BATCH_SIZE; r++) {
      int best = 0;
      for (int c = 1; c < OUTPUT_SIZE; c++)
        if (h_z2[r * OUTPUT_SIZE + c] > h_z2[r * OUTPUT_SIZE + best]) best = c;
      if (best == h_yb[r]) test_correct++;
    }
  }
  printf("---\ntest accuracy: %.2f%%\n", 100.0f * test_correct / (test_batches * BATCH_SIZE));

  cudaFree(nn.weights1); cudaFree(nn.bias1); cudaFree(nn.weights2); cudaFree(nn.bias2);
  cudaFree(nn.grad_weights1); cudaFree(nn.grad_bias1); cudaFree(nn.grad_weights2); cudaFree(nn.grad_bias2);
  cudaFree(z1); cudaFree(z2);
  cudaFree(dlogits); cudaFree(da1); cudaFree(dz1); cudaFree(dXin);
  free(h_z2); free(h_dlogits); free(h_yb);
  cudaFree(d_X); cudaFree(d_y); cudaFree(d_Xt); cudaFree(d_yt);
  
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
  return 0;
}