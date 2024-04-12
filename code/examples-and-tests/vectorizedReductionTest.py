import math
# import random
import time

# from os.path import exists as os_path_exists
# from urllib.request import urlopen
# from itertools import permutations

import numpy as np
# import pandas as pd
import numba

# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# from IPython.display import clear_output



def getCudaCores():
    '''Source of this code: https://stackoverflow.com/a/63833950'''

    CC_CORES_PER_SM = {
        (2,0): 32,
        (2,1): 48,
        (3,0): 192,
        (3,5): 192,
        (3,7): 192,
        (5,0): 128,
        (5,2): 128,
        (6,0): 64,
        (6,1): 128,
        (7,0): 64,
        (7,5): 64,
        (8,0): 64,
        (8,6): 128,
        (8,9): 128,
        (9,0): 128,
    }
    # the above dictionary should result in a value of "None" if a cc match 
    # is not found.  The dictionary needs to be extended as new devices become
    # available, and currently does not account for all Jetson devices

    device = numba.cuda.get_current_device()
    mySMS = getattr(device, 'MULTIPROCESSOR_COUNT')
    myCC = device.compute_capability
    coresPerSM = CC_CORES_PER_SM.get(myCC)
    cudaCores = coresPerSM*mySMS
    # print("GPU compute capability: " , my_cc)
    # print("GPU total number of SMs: " , my_sms)
    # print("total cores: " , CUDA_CORES)
    return cudaCores


@numba.guvectorize(
    ['void(float64[:], float64[:], float64[:])'],
    '(d),(d)->(d)',
    nopython=True,
    target='cuda'
)
def sumArrays(arr1:list[np.float64], arr2:list[np.float64], arrResult:list[np.float64]):
    d_ = len(arr1)
    for dimIdx in range(d_):
        arrResult[dimIdx] = (arr1[dimIdx] + arr2[dimIdx] + arrResult[dimIdx])



def sumGPUv1(arr:np.array, cores:int=-1, n_:int=-1, d_:int=-1):
    if n_ <= 0: n_ = len(arr)
    if d_ <= 0: d_ = len(arr[0])
    if cores <= 0: cores = getCudaCores()

    arrAcum = np.zeros((cores, d_), np.float64)

    offset = 0
    i = 1
    while cores*(offset+2) <= n_ - 1:
        # print(f'Iteration #{i}')
        # print(f'arr[{cores*offset}:{cores*(offset+1)}]')
        # print(f'arr[{cores*(offset+1)}:{cores*(offset+2)}]')

        # numba.cuda.synchronize()
        arrResult = np.zeros((cores, d_), np.float64)
        sumArrays(arr[cores*offset:cores*(offset+1)], arr[cores*(offset+1):cores*(offset+2)], arrResult)
        arrAcum += arrResult
        offset += 2
        i += 1

        # print(f'Sum so far: {arrAcum}')

        # print('\n')

    offset -= 2
    # print(f'i = {offset}')

    return np.sum(arrAcum, axis=0)

# @numba.cuda.reduce
# def sum_reduce(a, b):
#     return a + b
@numba.cuda.reduce
def sumGPUv2(dp1:np.array, dp2:np.array):
    return dp1 + dp2


if __name__ == '__main__':
    # CUDA_CORES = getCudaCores()
    CUDA_CORES = 5888
    print(f'Available CUDA cores: {CUDA_CORES}')

    N = CUDA_CORES * math.ceil(1e8 / CUDA_CORES)
    D = 3

    arr = np.full((N, D), 3.14159265, np.float64)

    # Numpy sum
    startTimeNS = time.perf_counter_ns()
    sum = np.sum(arr, axis=0)
    elapsedTimeS = (time.perf_counter_ns() - startTimeNS) * 1e-9
    print(f'Numpy | Sum = {sum}\nDone in {elapsedTimeS:.8f} s')

    # GPU sum v1
    startTimeNS = time.perf_counter_ns()
    sum = sumGPUv1(arr, CUDA_CORES, N, D)
    elapsedTimeS = (time.perf_counter_ns() - startTimeNS) * 1e-9
    print(f'sumGPUv1 | Sum = {sum}\n(len={len(sum)})\nDone in {elapsedTimeS:.8f} s')

    # GPU sum v2
    startTimeNS = time.perf_counter_ns()

    sum = np.zeros(D)
    for dimIdx in range(D):
        sum[dimIdx] = sumGPUv2(arr[:,dimIdx].copy())

    elapsedTimeS = (time.perf_counter_ns() - startTimeNS) * 1e-9

    print(f'sumGPUv2 | Sum = {sum}\n(len={len(sum)})\nDone in {elapsedTimeS:.8f} s')
