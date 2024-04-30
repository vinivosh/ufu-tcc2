#include <iostream>
#include <math.h>

#include <chrono>
using namespace std::chrono;



// Função que adiciona os elementos de dois vetores
void add(long n, float *x, float *y){
  for (long i = 0; i < n; i++)
    y[i] += x[i];
}

int main(void){
  long N = long(1<<28) + long(1<<27); // 402.653.184 elementos

  float *x = new float[N];
  float *y = new float[N];

  // Inicializar vetores no host
  for (long i = 0; i < N; i++) {
    x[i] = 3.77f; y[i] = 3.23f;
  }

  int runs = 100;
  auto startTime = high_resolution_clock::now();

  for (int i = 0; i < runs; i++) {
    // Rodar na CPU
    add(N, x, y);
  }

  auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - startTime);
  std::cout << "Avg exec time: " << duration.count() / runs << " ms" << "\n";

  // Checar se há erros (todos valores devem ser 3.23f + runs * 3.77f)
  float expectedSum = 3.23f + runs * 3.77f;
  float maxError = 0.0f;
  for (long i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - expectedSum));
  std::cout << "Max error: " << maxError << "\n";

  // Liberar memória
  delete [] x;
  delete [] y;

  return 0;
}