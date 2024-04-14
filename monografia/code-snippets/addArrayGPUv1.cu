#include <iostream>
#include <math.h>

// Kernel que adiciona os elementos de dois vetores
__global__
void add(int n, float *x, float *y){
  for (int i = 0; i < n; i++)
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

  for (int i = 0; i < 10; i++) {
    add<<<1, 1>>>(N, x, y);
    cudaDeviceSynchronize();
  }

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-7.0f));
  std::cout << "Max error: " << maxError << "\n";

  cudaFree(x);
  cudaFree(y);
  
  return 0;
}