#include <iostream>
#include <math.h>

// Função que adiciona os elementos de dois vetores
void add(int n, float *x, float *y){
  for (int i = 0; i < n; i++)
    y[i] += x[i];
}

int main(void){
  int N = 1<<28; // 268.435.456 elementos

  float *x = new float[N];
  float *y = new float[N];

  // Inicializar vetores no host
  for (int i = 0; i < N; i++) {
    x[i] = 3.77f; y[i] = 3.23f;
  }

  // Rodar na CPU
  add(N, x, y);

  // Checar se há erros (todos valores devem ser  7.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 7.0f));
  std::cout << "Max error: " << maxError << "\n";

  // Liberar memória
  delete [] x;
  delete [] y;

  return 0;
}