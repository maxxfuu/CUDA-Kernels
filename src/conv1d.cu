#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

void conv1d_cpu(const float* in, float* out, const float* kernel,
                int input_size, int kernel_size) {
  int output_size = input_size - kernel_size + 1;
  for (int i = 0; i < output_size; ++i) {
    float sum = 0.0f;
    for (int j = 0; j < kernel_size; ++j) {
      sum += in[i + j] * kernel[j];
    }
    out[i] = sum;
  }
}

__global__ void conv1d_kernel(const float* in, float* out,
                              const float* kernel, int input_size, int kernel_size) {
  int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int output_size = input_size - kernel_size + 1;
  if (output_idx < output_size) {
    float sum = 0.0f;
    for (int k_idx = 0; k_idx < kernel_size; ++k_idx) {
      sum += in[output_idx + k_idx] * kernel[k_idx];
    }
    out[output_idx] = sum;
  }
}

int main() {
  int input_size = 1000000, kernel_size = 32;
  int output_size = input_size - kernel_size + 1;

  size_t in_bytes = input_size * sizeof(float);
  size_t out_bytes = output_size * sizeof(float);
  size_t kernel_bytes = kernel_size * sizeof(float);

  float* h_in = (float*)malloc(in_bytes);
  float* h_kernel = (float*)malloc(kernel_bytes);
  float* h_out = (float*)malloc(out_bytes);
  float* h_out_cpu = (float*)malloc(out_bytes);

  float* d_in = nullptr;
  float* d_kernel = nullptr;
  float* d_out = nullptr;

  cudaMalloc(&d_in, in_bytes);
  cudaMalloc(&d_kernel, kernel_bytes);
  cudaMalloc(&d_out, out_bytes);

  for (int i = 0; i < input_size; i++) {
    h_in[i] = (float)(rand() % 10);
  }
  for (int i = 0; i < kernel_size; i++) {
    h_kernel[i] = (float)(rand() % 10);
  }

  cudaMemcpy(d_in, h_in, in_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(256);
  dim3 blocksPerGrid((output_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
  conv1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, d_kernel,
                                                    input_size, kernel_size);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaDeviceSynchronize();

  cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

  conv1d_cpu(h_in, h_out_cpu, h_kernel, input_size, kernel_size);

  int errors = 0;
  for (int i = 0; i < output_size; i++) {
    if (fabsf(h_out[i] - h_out_cpu[i]) > 1e-5f) {
      if (errors < 5) {
        printf("Mismatch at %d: GPU=%f CPU=%f\n", i, h_out[i], h_out_cpu[i]);
      }
      errors++;
    }
  }
  printf(errors == 0 ? "passed\n" : "failed\n");

  cudaFree(d_in);
  cudaFree(d_kernel);
  cudaFree(d_out);
  free(h_in);
  free(h_kernel);
  free(h_out);
  free(h_out_cpu);

  return 0;
}
