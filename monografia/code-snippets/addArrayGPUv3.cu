#include <iostream>
#include <math.h>

__global__
void add(long n, float *x, float *y){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (long i = index; i < n; i += stride)
    y[i] += x[i];
}

int main(void){
  long N = long(1<<28) + long(1<<27); // 402.653.184 elementos

  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  for (long i = 0; i < N; i++) {
    x[i] = 3.77f; y[i] = 3.23f;
  }

  int blockSize = 1024;
  int numBlocks = ceil(N / blockSize);

  add<<<numBlocks, blockSize>>>(N, x, y);
  cudaDeviceSynchronize();

  float maxError = 0.0f;
  for (long i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 7.0f));
  std::cout << "Max error: " << maxError << "\n";

  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
