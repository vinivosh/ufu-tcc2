import numpy as np; import pandas as pd

def kMeansCPU(dataset:pd.DataFrame, k=3, maxIter=100):
    centroids = pd.concat([(dataset.apply(lambda x: float(x.sample().iloc[0]))) for _ in range(k)], axis=1)
    centroids_OLD = pd.DataFrame()

    datasetLogs = np.log(dataset)

    iteration = 1

    while iteration <= maxIter and not centroids_OLD.equals(centroids):
        distances = centroids.apply(lambda x: np.sqrt(((dataset - x) ** 2).sum(axis=1)))
        closestCent = distances.idxmin(axis=1)
        del distances

        centroids_OLD = centroids
        centroids = datasetLogs.groupby(closestCent).apply(lambda x: np.exp(x.mean())).T

        iteration += 1

    return closestCent
