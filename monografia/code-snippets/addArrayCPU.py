import time; import numpy as np; import numba

def addArrayCPU(a, b):
    return a + b

def checkMaxErr(c):
    # Checando erro máximo (todos elementos devem ser 42.0):
    minRow = c.min(axis=0)
    maxRow = c.max(axis=0)
    maxErr = 0.0
    for dIdx in range(D):
        maxErr = max(maxErr, abs(42.0 - minRow[dIdx]))
        maxErr = max(maxErr, abs(42.0 - maxRow[dIdx]))
    print(f'Max error: {maxErr}')

D = 2**2
N = int(2**28 * 1.5) // D # N * D = 402.653.184 elementos

# Inicializando vetores
a = np.full((N, D), 27.2, np.float32)
b = np.full((N, D), 14.8, np.float32)

# Realizando adição
c = addArrayCPU(a, b)

checkMaxErr(c)



if __name__ == '__main__':
    RUNS = 100
    execTimes = np.zeros(RUNS, np.int64)

    print(f'Realizando benchmark de addArrayCPU (rodando {RUNS}x)...')

    for i in range(RUNS):
        startTime = time.perf_counter_ns()
        c = addArrayCPU(a, b)
        execTimes[i] = time.perf_counter_ns() - startTime

    checkMaxErr(c)

    # Calculando média e desvio padrão (e convertendo ns -> ms)
    execTimesMean = execTimes.mean() * 1e-9 * 1e3
    execTimesStd = execTimes.std() * 1e-9 * 1e3

    print(' \nexecTimes:')
    for i in range(RUNS): print(execTimes[i])
    print('\n')
    print(f'Benchmark concluído! Tempo médio: {execTimesMean:.6f} ± {execTimesStd:.6f} ms')
