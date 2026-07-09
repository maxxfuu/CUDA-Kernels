//include <cstdio>
//include <cuda_runtime.h>

__global__ void vectorAdd(float *a, float *b, float *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    // n is the number of elements in the array
    int n = 8;
    // bytes is the size of the array in bytes
    size_t bytes = n * sizeof(float);
    
    // allocate memory on the host for the arrays
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // initialize the arrays with values 
    for (int i = 0; i < n; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // allocate memory on the device for the arrays
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    // copy the arrays from the host to the device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // launch the kernel -> kernel code is defined under the global execution specifer 
    vectorAdd<<<1, 8>>>(d_a, d_b, d_c);

    // copy the results back into the device
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    int success = 1;
    for (int i = 0; i < n; ++i) {
        if (h_c[i] != (h_a[i] + h_b[i])) {
            printf("Error at index %d: Got %f, expected %f\n",
                   i, h_c[i], (h_a[i] + h_b[i]));
            success = 0;
            break;
        }
    }

    if (success) {
        printf("All elements are correct.\n");
    }

    // free the memory that was allocated on the host and device
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
