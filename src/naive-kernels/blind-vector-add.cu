#include <cstdio>
#include <cuda_runtime.h>

int main() { 
  // define the data 
  int n = 12;
  size_t bytes = n * sizeof(float);

  // declares a pointer variable called h_something of type float 
  // (float*) is a c style cast which casts the return type of the memoy allocation 
  // malloc(bytes) is to allocate bytes on the heap and reurns a pointer to teh first byte of that contiguous block  

  float *h_a = (float*)malloc(bytes);
  float *h_b = (float*)malloc(bytes);
  float *h_c = (float*)malloc(bytes);

  // initalize the array with values 
  for (int i = 0; i < n; i++) {
    h_a[i] = float(i);
    h_b[i] = float(i * 2);
  }

  // allocate memory memory on device, mirroring what we did on the host
  // d_something is a small variable in CPU RAM that points to a a separate block of GPU memory 
  // d_something is a pointer to a memory address with some data and treat it as a float 
  float *d_a, *d_a, *d_c;

  // (void**) - is a pointer to a pointer to void. 
  // d_a is variable that points to some data. It type is float*
  // points to the d_a variable itself, which points to the GPU address, It type is float**
  
  cudaMalloc((void**)&d_a, bytes);
  cudaMalloc((void**)&d_b, bytes);
  cudaMalloc((void**)&d_c, bytes);

  // copy the arrays from the host to the device 
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  // launch the kernel that is defined in under the global execution specifier 
  // specify the function name, followed by <<<grid, block>>>(arg1, arg2, arg3)

  // grid - is the grid size which defines the number of blocks to launch 
  // block - is the block size which defines the number of threads per block 
  // grid * block = total thread 

  // arg1 to arg 3 determines what data to use to perform SIMT
  vectoradd<<<1, 8>>>(d_a, d_b, d_c);

  // copy the results from the device back into the host 
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
  
  // free the memory that was allocated on the host and device
  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}