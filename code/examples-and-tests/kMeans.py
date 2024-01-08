from os.path import exists as os_path_exists
from urllib.request import urlopen
from itertools import permutations

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from IPython.display import clear_output



def downloadFileIfNeeded(filePath, url):
    '''Função auxiliar para baixar um arquivo do `url`, se ele não existir no caminho `filePath`'''
    if not os_path_exists(filePath):
        with urlopen(url) as f:
            html = f.read().decode('utf-8')
        with open(filePath, 'w') as f:
            f.write(html)


def kMeansCPU(dataset:pd.DataFrame, k=3, maxIter=100, plotResults=False, debug=False):
    if plotResults:
        # Inicializando variáveis para exibição gráfica
        pca = PCA(n_components=2) # dois eixos no gráfico
        dataset_2D = pca.fit_transform(dataset)

    # Gerando centróides iniciais randomicamente
    centroids = pd.concat([(dataset.apply(lambda x: float(x.sample().iloc[0]))) for _ in range(k)], axis=1) # * Paralelizar isto provavelmente é irrelevante, visto que sempre teremos poucos centróides
    centroids_OLD = pd.DataFrame()

    iteration = 1

    while iteration <= maxIter and not centroids_OLD.equals(centroids):
        if plotResults or debug: clear_output(wait=True)
        if debug: debugStr = f'Iteration {iteration}\n\nCentroids:\n{centroids.T}\n\n'

        # Para cada datapoint, calcular distâncias entre ele e cada centróide; depois, encontrar o centróide mais próximo e salvar seu index
        distances = centroids.apply(lambda x: np.sqrt(((dataset - x) ** 2).sum(axis=1))) # ! Parte altamente paralelizável!
        if debug: debugStr += f'Distances:\n{distances}\n\n'
        closestCent = distances.idxmin(axis=1)
        del distances
        if debug: debugStr += f'Closest centroid index:\n{closestCent}\n\n'

        centroids_OLD = centroids
        centroids = dataset.groupby(closestCent).apply(lambda x: np.exp(np.log(x).mean())).T # ! Parte altamente paralelizável!

        if plotResults:
            # Plotando clusters
            centroids_2D = pca.transform(centroids.T)
            plt.title(f'Iteration {iteration}')
            plt.scatter(x=dataset_2D[:,0], y=dataset_2D[:,1], c=closestCent)
            plt.scatter(x=centroids_2D[:,0], y=centroids_2D[:,1], marker='+', linewidths=2, color='red')
            plt.show()

        if debug: print(debugStr)

        iteration += 1

    return closestCent


def getClassificationHits(results:pd.DataFrame, dataset:pd.DataFrame, classColumn:str|int='class', classes:list[str]=None, debug=False):
    '''Função auxiliar que retorna uma tupla com três informações: (1) a quantidade de acertos de classificação expressos no dataframe `results`; (2) as quantidades destes acertos separadas por classes; e (3) a interpretação das classes usadas para encontrar o melhor resultado
    
    Para determinar acertos e erros, é usada a coluna de nome/index `classColumn` do dataframe `dataset` como fonte de verdade. O maximum matching é feito para encontrar a melhor interpretação dos resultados e usá-la como resultado final

    `results` deve ser um dataframe com duas colunas, a primeira sendo o index do datapoint e a segunda sendo o index da classe à qual o datapoint foi classificado
    
    Uma lista com os nomes das classes pode ser passada em `classes`, para agilizar o processo. Se nada for passado, as classes serão inferidas pela coluna de nome/index `classColumn` do dataframe `dataset`

    Exemplo de retorno: `(118, {'Iris-setosa': 35, 'Iris-versicolor': 39, 'Iris-virginica': 44}, ('Iris-virginica', 'Iris-setosa', 'Iris-versicolor'))`
    
    O retorno acima significa que (1) houveram `118` acertos totais, (2) sendo `35` desses da classe `Iris-setosa`, `39` da classe `Iris-versicolor` e `44` da classe `Iris-virginica` e (3) que a interpretação usada para obter este melhor resultado foi: classe `0` em `results` = `Iris-virginica`, classe `1` = `Iris-setosa` e  classe `2` = `Iris-versicolor`.'''

    def getHitsForClassAndPosition(class_:str, position:int, results:pd.DataFrame=results, dataset:pd.DataFrame=dataset, classColumn:str|int=classColumn):
        '''Função auxiliar interna que calcula os acertos de uma única interpretação de uma única classe
        
        Tal interpretação é expressa em duas variáveis: `class_` -> nome da classe (assim como está expressa na coluna `classColumn` do `dataset`) e `position` -> posição da classe quanto ao array das possíveis k classes ([0, …, k-1] da segunda coluna de `results`)
        
        Exemplo:
        
        `getHitsForClassAndPosition('Iris-setosa', 1)` irá calcular os acertos em `results` ao considerar a classificação de index `1` como simbolizando a classe `Iris-setosa`'''

        hits = 0

        print(f'Counting hits for class "{class_}" being index "{position}" in the results dataset...')

        for rowIndex in range(0, len(results)):
            row = results.iloc[[rowIndex]]
            # datapointIndex = row.index.values[0]
            resultClassIndex = row.values[0]

            # isCorrect = None
            # correctClass = None

            # Conferimos apenas linhas onde o index em "results" é o mesmo em "position", pois só nos importamos com acertos para palpites deste index
            if resultClassIndex == position:
                # Obtendo a classe correta da nossa "fonte de verdade", o dataset "dataset"
                correctClass = dataset.iloc[[rowIndex]][classColumn].values[0]

                # Checando se foi um acerto
                isCorrect = True if str(correctClass) == str(class_) else False
                if isCorrect: hits += 1
                
            # print(f'dpIndex = {datapointIndex}; result = {resultClassIndex}; rowDataset = {correctClass}; isCorrect = {isCorrect}')

        print(f'Total hits: {hits}\n')
        return hits
        return np.random.randint(0, 50 + 1)

    if debug: print('#################### Computing classification hits... ####################\n')

    if classes is None or len(classes) == 0:
        classes = list(dataset[classColumn].factorize()[1])
        if debug: print(f'classes (inferred from `dataset`):\n{classes}\n')

    # Foçar classes para strings
    classes = [str(c) for c in classes]

    # Encontrar todas as permutações possíveis das classes. A posição de uma classe na permutação será usada para atrelar uma classe ao centroid de mesmo index. Por exemplo, na permutação ('Iris-versicolor', 'Iris-setosa', 'Iris-virginica'), o centróide 0 corresponderá à classe 'Iris-versicolor', o centróide 1 à classe 'Iris-setosa' e o centróide 2 à classse 'Iris-virginica'. Cada datapoint terá sua classificação definida através dessa relação
    classPerm = list(permutations(classes))

    if debug:
        print(f'classesPermutatons (len={len(classPerm)}):\n')
        for perm in classPerm: print(perm)
        print('\n')

    # Dicionário auxiliar que guarda a quantidade de acertos por classe e posição na lista de classes. Isto economizará poder computacional, pois não repetiremos cálculos redundantes de hits de uma classe numa mesma posição.
    # Estrutura:
    # {
    #     'Iris-setosa/0': 50,
    #     'Iris-setosa/1': 0,
    #     '<class>/<positionInPermutation>': 45,…
    # }
    hitsByclassAndPosition = {}

    bestHits = -1
    bestHitsPerClass = None
    bestPerm = None

    # Para cada permutação das classes
    for permutation in classPerm:
        # print(f'permutation = {permutation}\n')
        hitsPerClass = dict.fromkeys(classes, -1)

        # Para cada classe na permutação atual
        for classPosition, class_ in enumerate(permutation):
            # Se os hits para essa classe nessa posição da permutação já foram computados, vamos usá-los. Se não, os computamos pela primeira vez e salvamos para usos posteriores
            classAndPositionStr = f'{class_}/{classPosition}' # Chave a ser usada no dicionário
            hits:int = hitsByclassAndPosition.get(classAndPositionStr, None)
            if hits is None:
                hits = getHitsForClassAndPosition(class_, classPosition)
                hitsByclassAndPosition[classAndPositionStr] = hits

            # Salvando a quantidade de acertos na classe atual, seja qual for
            hitsPerClass[class_] = hits
            # print(f'classPosition = {classPosition}; class_ = {class_}\nhitsPerClass = {hitsPerClass}\nhitsByclassAndPosition = {hitsByclassAndPosition}\n')

        # Computar acertos totais dessa permutação
        totalHits = sum(hitsPerClass.values())

        if totalHits > bestHits:
            bestHits = totalHits
            bestHitsPerClass = hitsPerClass
            bestPerm = permutation

        print(f'totalHits = {totalHits}\nbestHits = {bestHits}; bestHitsPerClass = {bestHitsPerClass}; bestPerm = {bestPerm}\n\n')

    return (bestHits, bestHitsPerClass, bestPerm)
