#
# ! ############################################################################
# ! Code taken and adapted from:
# ! https://towardsdatascience.com/cuda-by-numba-examples-215c0d285088
# ! All credits go to Carlos Costa, Ph.D.
# ! ############################################################################


import time
import numpy as np

import numba
from numba import cuda

threads_per_block_2d = (16, 16)  #  256 threads total
blocks_per_grid_2d = (64, 64)

# Total number of threads in a 2D block (has to be an int)
shared_array_len = int(np.prod(threads_per_block_2d))

# Example 2.4: 2D reduction with 1D shared array
@cuda.jit
def reduce2d(array2d, partial_reduction2d):
    ix, iy = cuda.grid(2)
    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2)

    s_thread = 0.0
    for i0 in range(iy, array2d.shape[0], threads_per_grid_x):
        for i1 in range(ix, array2d.shape[1], threads_per_grid_y):
            s_thread += array2d[i0, i1]

    # Allocate shared array
    s_block = cuda.shared.array(shared_array_len, numba.float32)

    # Index the threads linearly: each tid identifies a unique thread in the
    # 2D grid.
    tid = cuda.threadIdx.x + cuda.blockDim.x * cuda.threadIdx.y
    s_block[tid] = s_thread

    cuda.syncthreads()

    # We can use the same smart reduction algorithm by remembering that
    #     shared_array_len == blockDim.x * cuda.blockDim.y
    # So we just need to start our indexing accordingly.
    i = (cuda.blockDim.x * cuda.blockDim.y) // 2
    while (i != 0):
        if (tid < i):
            s_block[tid] += s_block[tid + i]
        cuda.syncthreads()
        i //= 2
    
    # Store reduction in a 2D array the same size as the 2D blocks
    if tid == 0:
        partial_reduction2d[cuda.blockIdx.x, cuda.blockIdx.y] = s_block[0]


N_2D = (36_000, 36_000)
a_2d = np.arange(np.prod(N_2D), dtype=np.float32).reshape(N_2D)
a_2d /= a_2d.sum() # a_2d will have sum = 1 (to float32 precision)

tic = time.perf_counter()
s_2d_cpu = a_2d.sum()
timing_2d_cpu = (time.perf_counter() - tic) * 1e3

print(f"Elapsed time cpu: {timing_2d_cpu:.0f} ms")

dev_a_2d = cuda.to_device(a_2d)
dev_partial_reduction_2d = cuda.device_array(blocks_per_grid_2d)

reduce2d[blocks_per_grid_2d, threads_per_block_2d](dev_a_2d, dev_partial_reduction_2d)
s_2d = dev_partial_reduction_2d.copy_to_host().sum()  # Final reduction in CPU

np.isclose(s_2d, s_2d_cpu)  # Ensure we have the right number
# True

timing_2d = np.empty(10)
for i in range(timing_2d.size):
    tic = time.perf_counter()
    reduce2d[blocks_per_grid_2d, threads_per_block_2d](dev_a_2d, dev_partial_reduction_2d)
    s_2d = dev_partial_reduction_2d.copy_to_host().sum()
    cuda.synchronize()
    toc = time.perf_counter()
    assert np.isclose(s_2d, s_2d_cpu)    
    timing_2d[i] = toc - tic
timing_2d *= 1e3  # convert to ms

print(f"Elapsed time better: {timing_2d.mean():.0f} ± {timing_2d.std():.0f} ms")
# Elapsed time better: 15 ± 4 ms