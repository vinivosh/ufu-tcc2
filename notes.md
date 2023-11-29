# Overview do progresso

## O que foi feito

- Implementação do K-Means CPU
- Implementação do K-Means GPU
- Realização de testes de corretude do K-means CPU
- Realização de testes de comparação de tempo de execução entre K-means CPU e GPU
- Escolher um outro algoritmo de agrupamento mais complexo e muito utilizado em grandes datasets para ser estudado e usado de exemplo → **Hierarchical Clustering**



## O que ainda tem que ser feito

- Rever introdução teórica
- Mudar o tempo verbal do TCC! Futuro → passado, ou seja, "Esta pesquisa irá focar em […]" → "Esta pesquisa focou em […]"
- Mudar "clusterização" para "agrupamento". Esse termo portugês oficial é melhor
- Escrever parte teórica explicando funcionamento do k-means
- Escrever parte teórica explicando funcionamento outro algoritmo de agrupamento
- Descrever implementação do K-Means GPU
- Descrever experimentos de comparação de tempo de execução entre K-means CPU e GPU
- Implementar outro algoritmo (CPU)
- Implementar outro algoritmo (GPU)
- Descrever implementação do outro algoritmo (GPU)
- Realizar experimentos de comparação de tempo de execução entre outro algoritmo CPU e GPU
- Descrever experimentos de "
- Descrever "receita geral de paralelização"
- Escrever conclusão





# Notas Misc.

## Links úteis

- [Trabalhos de Conclusão de Curso feitos por alunos da FACOM](https://repositorio.ufu.br/handle/123456789/5142/browse?type=type&order=ASC&rpp=20&value=Trabalho+de+Conclus%C3%A3o+de+Curso)
- [Datasets open-source que podem ser usados para testes e experimentos](https://archive.ics.uci.edu/datasets?NumInstances=1000-inf&NumFeatures=0-10&skip=0&take=10&sort=desc&orderBy=NumHits&search=&Types=Multivariate) (DSs daqui foram usados até no doutorado do Daniel Abdala!)



## Sanando Dúvidas de Português Muito Comuns

Essas me pegam desprevinido sempre! Bom manter salvo aqui pra referência rápida (e offline!).

### Os Quatro Porquês

**POR QUE** separado é o porquê de perguntas (diretas ou indiretas), o porquê que equivale a "por qual razão", e também tem o mesmo sentido de "pelo qual" e suas flexões.

- **Por que** eu tenho que aprender isto?
- Gostaria de saber **por que** eu tenho que aprender isto.
- **Por que** sobra sempre para mim?
  - (*Por qual razão* sobra sempre para mim?)
- A razão **por que** sobra sempre para mim, eu não sei.
  - (A razão *pela qual* sobra sempre para mim, eu não sei)

**PORQUE** junto é o porquê de respostas.

- Não preciso de mais exemplos **porque** já entendi.

**PORQUÊ** junto e com acento é o porquê que representa um substantivo (o porquê, o motivo).

- Acho que você já entendeu o **porquê** de aprender isto.

**POR QUÊ** separado e com acento é o o porquê do fim das frases (com ponto de interrogação, de exclamação ou com ponto final).

- Você entendeu, sabe **por quê**? Eu sei **por quê**!

> Fonte: [Toda Matéria](https://www.todamateria.com.br/uso-do-por-que-porque-por-que-e-porque/)



## Capítulo 2 — Fundamentação Teórica → O que deve conter?

Sugestão de outro algoritmo para estudarmos à fundo no trabalho: **Hierarchical Clustering**

Sub-capítulos **Necessários**:

- Agrupamento de Dados
- Programação Vetorial
- Cuda
- K-Means
- Hierarchical Clustering (?)
