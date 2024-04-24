# Script Para a Apresentação

Um simples script para planejamento do que será dito oralmente na apresentação. Uma versão bem menos resumida dos [slides](https://github.com/vinivosh/ufu-tcc2/blob/main/presentation/Apresenta%C3%A7%C3%A3o%20-%20TCC%202.odp), digamos.

## Anotações

O que deve haver na apresentação (em ordem)?

- Slide de título, com nomes meu, do orientador e dos professores da banca;
- Slides explicando a motivação do trabalho;
- Slide apresentando a estrutura da apresentação — como um sumário dos slides seguintes;
- Slides explicando CPUs vs. GPUs;
  - Baixo nível de paralelismo vs. alto nível de paralelismo;
  - Operações escalares vs. operações vetoriais;
- Slides explicando o que faz uma operação paralelizável ou não;
  - Dois exemplos de operações paralelizáveis;
  - Dois exemplos de operações não-paralelizáveis;
  - Explicando a lei de Amdahl;
- Slides explicando o que são algoritmos de agrupamento de dados em geral;
- Slides explicando, por cima, o funcionamento do k-means;
  - Mostrar o pseudocódigo do k-means, junto com uma exemplificação com dataset fictício unidimensional;
  - Mostrar a execução no dataset real, porém pequenino, Iris (usar plots de uma execução do kMeansGPU aqui);
  - Analisar quais são as partes paralelizáveis do k-means;
- Slides explicando a API CUDA da NVIDIA;
  - Mostrar exemplo de implementação serial de soma de vetores em C++;
  - Mostrar exemplo de implementação paralela de soma de vetores em C++ e CUDA;
  - Comparar o speed-up;
- Slides explicando a biblioteca Numba;
  - Mostrar exemplo de implementação serial de soma de vetores 2D em Python;
  - Mostrar exemplo de implementação paralela de soma de vetores 2D em Python e Numba;
  - Comparar o speed-up;
  - Comparar a facilidade de desenvolvimento;
- Slides mostrando um pouco do código em CPU vs. GPU do k-means implementado;
  - Mostrar apenas um snippet do código (cálculo de distâncias?) e comparar como foi implementado;
- Slides mostrando os datasets e resultados;
  - Mostrar os datasets;
  - Mostrar os resultados de teste de speed-up;
  - Idem mas para os testes de corretude;
- Slides mostrando a conclusão do trabalho.




## Falas por Slide

O que será dito a cada slide, especificamente.



### Slide 1

[TBA]



### Slide 2

[TBA]



### Slide 3

[TBA]



### Slide 4

[TBA]



### Slide 5

[TBA]



### Slide 6

[TBA]



### Slide 7

[TBA]



### Slide 8

[TBA]



### Slide 9

[TBA]



### Slide 10

[TBA]

