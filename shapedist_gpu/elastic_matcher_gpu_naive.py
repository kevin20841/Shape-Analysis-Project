from numba import cuda, float32
import numpy as np


def elastic_gpu_optimized_single(t, p, q):
    n = t.size
    # Load data
    min_energy_values = np.full((n, n), np.inf, dtype=np.float32)
    min_energy_values[0][0] = 0
    flag1 = np.zeros((n, n), dtype=np.float32)
    flag2 = np.zeros((n, n), dtype=np.float32)
    t_global_mem = cuda.to_device(t)
    p_global_mem = cuda.to_device(p)
    q_global_mem = cuda.to_device(q)
    flag1_global_mem = cuda.to_device(flag1)
    flag2_global_mem = cuda.to_device(flag1)

    min_energy_values_global_mem = cuda.device_array(n)

    # Set the number of threads in a block
    threadsperblock = (32, 32)

    # Calculate the number of thread blocks in the grid
    blockspergrid = ((t.size + (32 - 1)) // 32, (t.size + (32 - 1)) // 32)
    # blockspergrid = (1, 1)
    # relax_edges
    for i in range(2*n):
        relax[blockspergrid, threadsperblock](t, p, q, flag1, flag2, min_energy_values)

    # Print the result
    min_energy_values = min_energy_values_global_mem.copy_to_host()
    print(min_energy_values)


@cuda.jit
def relax(t, p, q, flag1, flag2, min_energy_values):
    n = t.size
    i, j = cuda.grid(2)
    k, l = i - 32, j-32
    if k <= 0:
        k = 1
    if l <= 0:
        l = 1
    t_cache = cuda.shared.array(32, dtype=float32)
    p_cache = cuda.shared.array(32, dtype=float32)
    q_cache = cuda.shared.array(32, dtype=float32)
    e = 0
    temp = np.inf
    if i < t.size and j < t.size:  # Check array boundaries
        return
    for index1 in range(i - k):
        # Shared Memory
        t[index1] = t[k + i]
        p_cache[index1] = p[k + i]
        q_cache[index1] = q[k + i]

        # Wait for threads to finish preloading
        cuda.syncthreads()

        for index2 in range(j - l):
            e = min_energy_values[k, l] + integrate(t_cache, t_cache, p_cache, q_cache,
                                                    index1, i-k, index2, j-l, n)
            if e < temp:
                temp = e
        # Wait for threads to finish computation
        cuda.syncthreads()
    min_energy_values[i][j] = temp


@cuda.jit(device=True)
def interp(t, x, y, lower, upper):
    i = 0
    while lower < upper:
        i = lower + (upper - lower) // 2
        val = x[i]
        if t == val:
            break
        elif t > val:
            if lower == i:
                break
            lower = i
        elif t < val:
            upper = i

    if i == x.size-1:
        temp = y[i]
    else:
        temp = (t - x[i]) * (y[i + 1] - y[i]) / (x[i + 1] - x[i]) + y[i]
    return temp


@cuda.jit(device=True)
def integrate(tp, tq, py, qy, k, i, l, j, gamma_interval):
    e = 0
    a = k
    while a < i:
        gammak_1 = gamma_interval * l + (tp[a] - tp[k]) * \
                                        (gamma_interval * j - gamma_interval * l) / (tp[i] - tp[k])
        gammak_2 = gamma_interval * l + (tp[a+1] - tp[k]) * (gamma_interval * j - gamma_interval * l) \
                                        / (tp[i] - tp[k])
        e = e + (0.5 * (py[a] - interp(gammak_1, tq, qy, 0, tq.size)) ** 2
                         + 0.5 * (py[(a+1)] - interp(gammak_2, tq, qy, 0, tq.size)) ** 2) * \
                (tp[a+1] - tp[a]) * 0.5
        a = a + 1
    return e
