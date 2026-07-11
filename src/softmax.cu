#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

void softmax_cpu(const float* in, float* out, int num_rows, int num_cols) {
    for (int row = 0; row < num_rows; ++row) {
        float max_val = in[row * num_cols];
        for (int col = 1; col < num_cols; ++col) {
            if (in[row * num_cols + col] > max_val) {
                max_val = in[row * num_cols + col];
            }
        }
        float sum_exp = 0.0f;
        for (int col = 0; col < num_cols; ++col) {
            sum_exp += expf(in[row * num_cols + col] - max_val);
        }
        for (int col = 0; col < num_cols; ++col) {
            out[row * num_cols + col] = expf(in[row * num_cols + col] - max_val) / sum_exp;
        }
    }
}

// naive version: one thread handles an entire row
__global__ void softmax_kernel(const float* in, float* out, int num_rows, int num_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        float max_val = in[row * num_cols];
        for (int col = 1; col < num_cols; ++col) {
            if (in[row * num_cols + col] > max_val) {
                max_val = in[row * num_cols + col];
            }
        }
        float sum_exp = 0.0f;
        for (int col = 0; col < num_cols; ++col) {
            sum_exp += expf(in[row * num_cols + col] - max_val);
        }
        for (int col = 0; col < num_cols; ++col) {
            out[row * num_cols + col] = expf(in[row * num_cols + col] - max_val) / sum_exp;
        }
    }
}

int main() {
    int num_rows = 1024;
    int num_cols = 1024;

    size_t size = num_rows * num_cols * sizeof(float);

    // allocate host (CPU) memory
    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);
    float* h_out_cpu = (float*)malloc(size);

    // allocate device (GPU) memory
    float* d_in = nullptr;
    float* d_out = nullptr;

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // initialize host data
    for (int i = 0; i < num_rows * num_cols; i++) {
        h_in[i] = (float)(rand() % 10);
    }

    // copy input to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // launch kernel: one thread per row
    int block = 256;
    int grid = (num_rows + block - 1) / block;
    softmax_kernel<<<grid, block>>>(d_in, d_out, num_rows, num_cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    // copy result back to host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // verify against CPU reference
    softmax_cpu(h_in, h_out_cpu, num_rows, num_cols);

    int errors = 0;
    for (int i = 0; i < num_rows * num_cols; i++) {
        float diff = fabsf(h_out[i] - h_out_cpu[i]);
        if (diff > 1e-5f) {
            if (errors < 5) {
                printf("Mismatch at %d: GPU=%f CPU=%f\n", i, h_out[i], h_out_cpu[i]);
            }
            errors++;
        }
    }
    printf(errors == 0 ? "passed\n" : "failed\n");

    // free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    // free host memory
    free(h_in);
    free(h_out);
    free(h_out_cpu);

    return 0;
}
