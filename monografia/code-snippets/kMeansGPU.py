import math; import numpy as np; import pandas as pd; import numba

@numba.guvectorize(
    ['void(float64[:,:], float64[:], float64[:])'],
    '(k,d),(d)->(k)', nopython=True, target='cuda'
)
def calcDistances(centroids:list[list[np.float64]], rowDataset:list[np.float64], rowResults:list[np.float64]):
    d = len(rowDataset)

    for centroidIndex, centroid in enumerate(centroids):
        distance = 0.0
        for dim in range(d): distance += (rowDataset[dim] - centroid[dim]) ** 2
        distance = distance ** (1/2)

        rowResults[centroidIndex] = distance


@numba.guvectorize(
    ['void(float64[:],int64[:])'],
    '(k)->()', nopython=True, target='cuda'
)
def calcClosestCentroids(rowDistances:list[np.float64], closestCent:np.int64):
    minDistance = rowDistances[0]
    minDistanceIndex = 0

    for index, distance in enumerate(rowDistances):
        if distance < minDistance:
            minDistance = distance
            minDistanceIndex = index

    closestCent[0] = minDistanceIndex


@numba.guvectorize(
    ['void(float64[:],float64[:])'],
    '(d)->(d)', nopython=True, target='cuda'
)
def calcLogs(rowDataset:list[np.float64], rowResults:list[np.float64]):
    for dimIdx, dimValue in enumerate(rowDataset): rowResults[dimIdx] = math.log(dimValue)


def kMeansGPU(dataset:pd.DataFrame, k=3, maxIter=100):
    n = len(dataset)
    d = len(dataset.iloc[0])

    centroids:pd.DataFrame = pd.concat([(dataset.apply(lambda x: float(x.sample().iloc[0]))) for _ in range(k)], axis=1)
    centroids_OLD = pd.DataFrame()

    centroids__np = centroids.T.to_numpy()
    centroids_OLD__np = centroids_OLD.T.to_numpy()
    dataset__np = dataset.to_numpy()
    del dataset

    datasetLogs = np.zeros((n, d))
    calcLogs(dataset__np, datasetLogs)

    iteration = 1

    while iteration <= maxIter and not np.array_equal(centroids_OLD__np ,centroids__np):
        distances = np.zeros((n, k))
        calcDistances(centroids__np, dataset__np, distances)

        closestCent = np.zeros(n, np.int64)
        calcClosestCentroids(distances, closestCent)
        del distances

        centroids_OLD__np = centroids__np.copy()

        meansByClosestCent = np.zeros((k, d))

        for centroidIdx in range(k):
            x = [(True if closestCent[dpIdx] == centroidIdx else False) for dpIdx in range(n)]
            relevantLogs = datasetLogs[x]
            del x
            if len(relevantLogs) == 0: continue
            meansByClosestCent[centroidIdx] = relevantLogs.mean(axis=0)
            del relevantLogs

            centroids__np[centroidIdx] = np.exp(meansByClosestCent[centroidIdx])

        del meansByClosestCent
        iteration += 1

    return closestCent
