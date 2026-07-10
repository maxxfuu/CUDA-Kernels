#include <cstdio> 
#include <cuda_runtime.h> 

void transpose_matrix_cpu(float* in, float* out, int num_rows, int num_cols) {
  for (int row = 0; row < num_rows; row++) {
    for (int col = 0; col < num_cols; col++) {
      out[col * num_rows + row] = in[row * num_cols + col];
    }
  }
}

__global__ void transpose_matrix_gpu(float* in, float* out, int num_rows, int num_cols) {
  // defining global indexes
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (row < num_rows && col < num_cols) {
    out[col * num_rows + row] = in[row * num_cols + col];
  }
}

int main() {
  int num_rows = 28;
  int num_cols = 28;
  int num_batches = 100;
  size_t matrix_size = num_rows * num_cols * sizeof(float);
  size_t total_size = num_batches * matrix_size;

  float* h_in = (float*)malloc(total_size);
  float* h_out = (float*)malloc(total_size);

  for (int i = 0; i < num_batches * num_rows * num_cols; ++i) {
    h_in[i] = static_cast<float>(i % 100);
  }

  float *device_input, *device_output;
  cudaMalloc((void**)&device_input, total_size);
  cudaMalloc((void**)&device_output, total_size);

  cudaMemcpy(device_input, h_in, total_size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid(
    (num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y
  );

  for (int batch = 0; batch < num_batches; ++batch) {
    float* d_in = device_input + batch * num_rows * num_cols;
    float* d_out = device_output + batch * num_rows * num_cols;

    transpose_matrix_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, num_rows, num_cols);
  }

  cudaMemcpy(h_out, device_output, total_size, cudaMemcpyDeviceToHost);

  free(h_in);
  free(h_out);
  cudaFree(device_input);
  cudaFree(device_output);

  return 0;
}