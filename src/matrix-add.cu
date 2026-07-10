#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

// canonical truth, matrix addition w/ CPU
void matrixAdd(const float *a, const float *b, float *c, int num_rows, int num_cols) {
  for (int row = 0; row < num_rows; row++) {
    for (int col = 0; col < num_cols; col++) {
      // flatten 2d matrix into 1d
      int index = row * num_cols + col;
      c[index] = a[index] + b[index];
    }
  }
}

// kernel matrix addition w/ GPU
__global__ void matrixAdd_kernel(const float *a, const float *b, float *c, int num_rows, int num_cols) {

  // map x-dimension to columns and y-dimension to rows for coalesced access
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < num_rows && col < num_cols) {
    int index = row * num_cols + col;
    c[index] = a[index] + b[index];
  }
}

int main() {
  const int num_rows = 1000;
  const int num_cols = 500;
  size_t size = num_rows * num_cols * sizeof(float);

  // allocate host memory
  float *h_a = (float *)malloc(size);
  float *h_b = (float *)malloc(size);
  float *h_c_cpu = (float *)malloc(size);
  float *h_c_gpu = (float *)malloc(size);

  // initialize host arrays
  for (int i = 0; i < num_rows * num_cols; i++) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(i * 2);
  }

  // allocate device memory
  float *d_a, *d_b, *d_c;
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // copy data from host to device
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  // define block and grid dimensions
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // launch kernel
  matrixAdd_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, num_rows, num_cols);

  // copy results back to host
  cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);

  // verify correctness using CPU execution
  matrixAdd(h_a, h_b, h_c_cpu, num_rows, num_cols);

  bool success = true;
  for (int i = 0; i < num_rows * num_cols; i++) {
    if (std::abs(h_c_gpu[i] - h_c_cpu[i]) > 1e-5) {
      printf("Mismatch at index %d: CPU=%f, GPU=%f\n", i, h_c_cpu[i],
             h_c_gpu[i]);
      success = false;
      break;
    }
  }

  if (success) {
    printf("Verification succeeded! GPU output matches CPU canonical truth.\n");
  }

  // free allocated memory
  free(h_a);
  free(h_b);
  free(h_c_cpu);
  free(h_c_gpu);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}