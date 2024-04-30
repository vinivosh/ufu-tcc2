#include <iostream>
#include <math.h>

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

  // Rodar na CPU
  add(N, x, y);

  // Checar se há erros (todos os valores devem ser 7.0)
  float maxError = 0.0f;
  for (long i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 7.0f));
  std::cout << "Max error: " << maxError << "\n";

  // Liberar memória
  delete [] x;
  delete [] y;

  return 0;
}
