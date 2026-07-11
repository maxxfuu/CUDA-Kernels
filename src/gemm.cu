#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

void gemm_cpu(const float* A, const float* B, float* C, int M_rows, int N_cols, int K_shared_dim) {
  for (int row = 0; row < M_rows; row++) {
    for (int col = 0; col < N_cols; col++) {
      float sum = 0.0f;

      for (int k_idx = 0; k_idx < K_shared_dim; k_idx++) {
        sum += A[row * K_shared_dim + k_idx]
        * B[k_idx * N_cols + col];
      }
      C[row * N_cols + col] = sum;
    }
  } 
} 

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M_rows, int N_cols, int K_shared_dim) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M_rows && col < N_cols) {
    float sum = 0.0f; 
    for (int k_idx = 0; k_idx < K_shared_dim; ++k_idx) {
      sum += A[row * K_shared_dim + k_idx] * B[k_idx * N_cols + col];
    }
    C[row * N_cols + col] = sum;
  }
}

int main() {
  // define matrix dimensions
  int M_rows = 512;
  int N_cols = 512;
  int K_shared_dim = 512;

  size_t size_A = M_rows * K_shared_dim * sizeof(float);
  size_t size_B = K_shared_dim * N_cols * sizeof(float);
  size_t size_C = M_rows * N_cols * sizeof(float);

  // allocate host (CPU) memory
  float* h_A = (float*)malloc(size_A);
  float* h_B = (float*)malloc(size_B);
  float* h_C = (float*)malloc(size_C);
  float* h_C_cpu = (float*)malloc(size_C);

  // allocate device (GPU) memory
  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;

  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_B, size_B);
  cudaMalloc(&d_C, size_C);

  // initialize host data
  for (int i = 0; i < M_rows * K_shared_dim; i++) {
    h_A[i] = (float)(rand() % 10);
  }
  for (int i = 0; i < K_shared_dim * N_cols; i++) {
    h_B[i] = (float)(rand() % 10);
  }

  // copy input matrices to device
  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

  // launch kernel: x covers columns, y covers rows
  dim3 block(16, 16);
  dim3 grid((N_cols + block.x - 1) / block.x,
            (M_rows + block.y - 1) / block.y);
  gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M_rows, N_cols, K_shared_dim);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaDeviceSynchronize();

  // copy result back to host
  cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

  // verify against CPU reference
  gemm_cpu(h_A, h_B, h_C_cpu, M_rows, N_cols, K_shared_dim);

  int errors = 0;
  for (int i = 0; i < M_rows * N_cols; i++) {
    float diff = fabsf(h_C[i] - h_C_cpu[i]);
    // tolerance scales with magnitude since K=512 sums accumulate rounding error
    if (diff > 1e-3f * fabsf(h_C_cpu[i]) + 1e-3f) {
      if (errors < 5) {
        printf("Mismatch at %d: GPU=%f CPU=%f\n", i, h_C[i], h_C_cpu[i]);
      }
      errors++;
    }
  }
  printf(errors == 0 ? "passed\n" : "failed\n");

  // Free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_cpu);

  return 0;
}