#include <iostream>
#include <math.h>

#include <chrono>
using namespace std::chrono;

// Kernel que adiciona os elementos de dois vetores
__global__
void add(int n, float *x, float *y) {
  int index = threadIdx.x;
  int stride = blockDim.x;

  for (int i = index; i < n; i += stride)
    y[i] += x[i];
}

int main(void){
  int N = 1<<28; // 268.435.456 elementos

  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 3.77f; y[i] = 3.23f;
  }

  int runs = 10;
  auto startTime = high_resolution_clock::now();

  for (int i = 0; i < runs; i++) {
    add<<<1, 1024>>>(N, x, y);
    cudaDeviceSynchronize();
  }

  auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - startTime);
  std::cout << "Avg exec time: " << duration.count() / runs << " ms" << "\n";

  float expectedSum = 3.23f + runs * 3.77f;
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - expectedSum));
  std::cout << "Max error: " << maxError << "\n";

  cudaFree(x);
  cudaFree(y);
  
  return 0;
}