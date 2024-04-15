import time; import numpy as np; import numba

@numba.guvectorize(
    ['void(float32[:],float32[:],float32[:])'],
    '(d),(d)->(d)', nopython=True, target='cuda'
)
def addArrayGPU(a, b, c):
    d = len(a)
    for dIdx in range(d):
        c[dIdx] = a[dIdx] + b[dIdx]

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
N = int(2**28 * 1.5) // D # N * D = 268.435.456 elementos

# Inicializando vetores
a = np.full((N, D), 27.2, np.float32)
b = np.full((N, D), 14.8, np.float32)

# Inicializando vetor de retorno
c = np.zeros((N, D), np.float32)

# # Realizando adição
addArrayGPU(a, b, c)

checkMaxErr(c)



if __name__ == '__main__':
    RUNS = 15
    execTimes = np.zeros(RUNS, np.float64)

    print(f'Realizando benchmark de addArrayGPU (rodando {RUNS}x)...')

    c = np.zeros((N, D), np.float32)
    for i in range(RUNS):
        startTime = time.perf_counter_ns()
        addArrayGPU(a, b, c)
        execTimes[i] = time.perf_counter_ns() - startTime

    checkMaxErr(c)

    execTimes *= 1e-9 * 1e3 # Convertendo tempos para ms

    print(f'Benchmark concluído! Tempo médio: {execTimes.mean():.6f} ± {execTimes.std():.6f} ms')
