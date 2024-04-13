import math
# import random
import time

# from os.path import exists as os_path_exists
# from urllib.request import urlopen
# from itertools import permutations

import numpy as np
# import pandas as pd
import numba
import nvidia_smi # nvidia-ml-py3 <- package name

# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# from IPython.display import clear_output



def getCudaHWinfo():
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

    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here. if there's more than one NVIDIA GPU, then this would't quite work

    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    totalVRAM = info.total
    freeVRAM = info.free

    totalVRAMpercudaCore = math.floor(totalVRAM / mySMS / coresPerSM)
    availableVRAMpercudaCore = math.floor(freeVRAM / mySMS / coresPerSM)

    nvidia_smi.nvmlShutdown()
    return totalVRAM, freeVRAM, cudaCores, totalVRAMpercudaCore, availableVRAMpercudaCore


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
    if cores <= 0: _, _, cores, _, _ = getCudaHWinfo()

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


@numba.guvectorize(
    ['void(float64[:,:], float64[:,:], float64[:])'],
    '(n,d),(n,d)->(d)',
    nopython=True,
    target='cuda'
)
def sum2DArrays(arr1:list[list[np.float64]], arr2:list[list[np.float64]], result:list[np.float64]):
    # Plain Python
    n_ = len(arr1)
    d_ = len(arr1[0])
    # sum = np.zeros(d_)
    for rowIdx in range(n_):
        for dimIdx in range(d_): result[dimIdx] = result[dimIdx] + arr1[rowIdx][dimIdx] + arr2[rowIdx][dimIdx]

    # for dimIdx in range(d_): result[dimIdx] = sum[dimIdx]

    # # Numpy
    # result = np.sum(arr1) + np.sum(arr2)


def sumGPUv3(arr:np.ndarray, n_:int=-1, d_:int=-1, dataType=np.float64, vramPerCCfree:int=-1, vramPercent:int=0.90):
    if n_ <= 0: n_ = len(arr)
    if d_ <= 0: d_ = len(arr[0])

    # print(f'n_ = {n_}; d_ = {d_}')

    if vramPerCCfree <= 0: _, _, _, _, vramPerCCfree = getCudaHWinfo()
    sizeOfRow = d_ * np.dtype(dataType).itemsize
    b_ = math.floor((vramPerCCfree * vramPercent) / (sizeOfRow * 2) + 1)

    if n_ % (b_* 2) == 0:
        totalRowsToPad = 0
    else:
        totalRowsToPad = (math.ceil(n_ / (b_ * 2)) * (b_ * 2)) - n_

    # print(f'Size of each block: {b_} rows\nSo, using {vramPercent * 100}% of the VRAM per CUDA thread, there will be two arrays of {b_} rows + 1 array with 1 row for the result in each block of data. Row size = {sizeOfRow} B)\nExceeding rows (number of zero-valued rows to be padded) = {totalRowsToPad}')
    del sizeOfRow

    if totalRowsToPad > 0:
        # Padding needed, can't divide de array into b_ * 2 rows
        # This command will basically add totalRowsToPad zero-valued rows to the array
        # print(f'len(arr) before padding = {len(arr)}')
        # print(f'arr before padding:\n{arr}')
        arr = np.pad(arr, [(0, totalRowsToPad), (0, 0)])
        del totalRowsToPad
        # print(f'len(arr) after padding = {len(arr)}')
        # print(f'arr after padding:\n{arr}')

    # print(f'len(arr) before split = {len(arr)}')
    # print(f'arr before split:\n{arr}')
    arr = np.split(arr, int(len(arr) / b_))
    lenArr = len(arr)
    # print(f'lenArr after split = {lenArr}')
    # print(f'arr[0] after split:\n{arr[0]}')
    # print(f'len(arr[0])= {len(arr[0])}')
    # print(f'arr[-2] after split:\n{arr[-2]}')
    # print(f'len(arr[-2])= {len(arr[-2])}')
    # print(f'arr[-1] after split:\n{arr[-1]}')
    # print(f'len(arr[-1])= {len(arr[-1])}')

    arrMidIdx = int(lenArr / 2) # An index such that arr[0:arrMidIdx] and arr[arrMidIdx:] will yield the first half of arr and the last half of arr perfectly
    # print(len(arr[:arrMidIdx]), len(arr[arrMidIdx:]))
    resultArr = np.zeros((arrMidIdx, d_), dataType)
    sum2DArrays(arr[:arrMidIdx], arr[arrMidIdx:], resultArr)

    return np.sum(resultArr, axis=0)




if __name__ == '__main__':
    VRAM_TOTAL, VRAM_FREE, CUDA_CORES, VRAM_PER_CUDA_CORE_TOTAL, VRAM_PER_CUDA_CORE_FREE = getCudaHWinfo()
    # CUDA_CORES = 5888
    print(f'Total VRAM: {(VRAM_TOTAL / 1024**2):.0f} MB; Free VRAM: {(VRAM_FREE / 1024**2):.2f} MB; Available CUDA cores: {CUDA_CORES} (estimated free / total VRAM per core: {(VRAM_PER_CUDA_CORE_FREE / 1024):.2f} KB / {(VRAM_PER_CUDA_CORE_TOTAL / 1024):.2f} KB)')

    # N = int(7e7)
    N = 13_932_632
    D = 3
    K = 7
    I = 10

    TIMES_EXECUTED = K * I

    arr = np.full((N, D), 3.14159265, np.float64)

    # Numpy sum
    startTimeNS = time.perf_counter_ns()
    for _ in range(TIMES_EXECUTED): sum = np.sum(arr, axis=0)
    elapsedTimeS = (time.perf_counter_ns() - startTimeNS) * 1e-9
    print(f'Numpy | Sum = {sum}\nDone in {elapsedTimeS:.8f} s\n\n')

    # GPU sum v1
    startTimeNS = time.perf_counter_ns()
    for _ in range(TIMES_EXECUTED): sum = sumGPUv1(arr, CUDA_CORES, N, D)
    elapsedTimeS = (time.perf_counter_ns() - startTimeNS) * 1e-9
    print(f'sumGPUv1 | Sum = {sum}\n(len={len(sum)})\nDone in {elapsedTimeS:.8f} s\n\n')

    # GPU sum v2
    startTimeNS = time.perf_counter_ns()

    for _ in range(TIMES_EXECUTED): 
        sum = np.zeros(D)
        for dimIdx in range(D):
            sum[dimIdx] = sumGPUv2(np.ascontiguousarray(arr[:,dimIdx]))

    elapsedTimeS = (time.perf_counter_ns() - startTimeNS) * 1e-9

    print(f'sumGPUv2 | Sum = {sum}\n(len={len(sum)})\nDone in {elapsedTimeS:.8f} s\n\n')
    
    # GPU sum v3
    startTimeNS = time.perf_counter_ns()
    # sum = np.zeros(D)
    for _ in range(TIMES_EXECUTED): sum = sumGPUv3(arr, N, D, vramPerCCfree=VRAM_PER_CUDA_CORE_TOTAL, vramPercent=0.9)
    elapsedTimeS = (time.perf_counter_ns() - startTimeNS) * 1e-9

    print(f'sumGPUv3 | Sum = {sum}\n(len={len(sum)})\nDone in {elapsedTimeS:.8f} s')
