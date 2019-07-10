from numba import cuda, float32, jit, SmartArray
import numpy as np
np.set_printoptions(threshold=np.nan, linewidth=400)
width = 4
TPB = 16
cache_shape = width + TPB


@jit()
def elastic_gpu_optimized_1D(p, q):
    n = p.size
    # Load data
    min_energy_values = np.full((n, n), np.inf, dtype=np.float32)
    path_nodes = np.zeros((n, n, 2), dtype=np.int16)
    min_energy_values[1][1] = integrate(p, q, 0, 1, 0, 1, 1/(n-1), 0, 0)
    p = SmartArray(p)
    q = SmartArray(q)
    min_energy_values = SmartArray(min_energy_values)
    path_nodes = SmartArray(path_nodes)
    m = SmartArray(np.full(1, 0, dtype=np.float32))
    # t_global_mem = cuda.const.array_like(t)
    # p_global_mem = cuda.const.array_like(p)
    # q_global_mem = cuda.const.array_like(q)
    # flag1_global_mem = cuda.to_device(flag1)
    # flag2_global_mem = cuda.to_device(flag2)

    # Set the number of threads in a block
    threadsperblock = (TPB, TPB)

    # Calculate the number of thread blocks in the grid
    blockspergrid = ((p.size + (threadsperblock[0] - 1)) // threadsperblock[0],
                     (p.size + (threadsperblock[1] - 1)) // threadsperblock[1])
    # blockspergrid = (1, 1)
    # relax_edges
    prev = np.inf
    for i in range(2 * n - 3):
        relax[blockspergrid, threadsperblock](p, q, min_energy_values, m)
        if prev != np.inf and prev == m[0]:
            break
        prev = m[0]
        # print(min_energy_values.__array__())
    relax_final[blockspergrid, threadsperblock](p, q, min_energy_values, path_nodes)
    # Print the result
    gamma_interval = 1/(n-1)

    i = n - 1
    j = n - 1
    min_energy_values[i][j] = integrate(p, q, 0, i, 0, j, gamma_interval, 0, 0)
    k = i - width
    if k <= 0:
        k = 0
    minimum = min_energy_values[i][j]
    while k < i:
        l = j - width
        if l <= 0:
            l = 0
        while l < j:
            e = min_energy_values[k, l] + integrate(p, q, k, i, l, j, gamma_interval, 0, 0)
            if e < minimum:
                minimum = e
                path_nodes[i][j][0] = k
                path_nodes[i][j][1] = l

            l = l + 1
        k = k + 1

    min_energy_values[i][j] = minimum
    path = np.zeros(n, dtype=np.float64)

    # !! Interpolate
    path_indices = np.zeros((n, 2), dtype=np.int16)
    path_indices[0][0] = n - 1
    path_indices[0][1] = n - 1

    i = 0
    while path_indices[i][0] != 0 or path_indices[i][1] != 0 and i + 1 < path.size:
        result = path_nodes[path_indices[i][0]][path_indices[i][1]]
        path_indices[i + 1][0] = result[0]
        path_indices[i + 1][1] = result[1]
        i = i + 1
    i = 0
    previous = 1
    previousIndex_domain = n - 1
    previousIndex_gamma = n - 1

    path[path_indices[0][0]] = gamma_interval * path_indices[0][1]
    while i < path_indices.size // 2 and previousIndex_domain != 0:
        path[path_indices[i][0]] = gamma_interval * path_indices[i][1]
        if previousIndex_domain - path_indices[i][0] > 1:
            j = 0
            val = (gamma_interval * (previousIndex_gamma - path_indices[i][1])) / \
                  (gamma_interval * previousIndex_domain - gamma_interval * path_indices[i][0])
            while j < previousIndex_domain - path_indices[i][0]:
                path[previousIndex_domain - j] = previous - (gamma_interval * previousIndex_domain -
                                                             gamma_interval * (previousIndex_domain - j)) * val

                j = j + 1
        previousIndex_domain = path_indices[i][0]
        previousIndex_gamma = path_indices[i][1]
        previous = gamma_interval * path_indices[i][1]
        i = i + 1
    return path


@cuda.jit
def relax(p, q, min_energy_values, m):
    p_cache = cuda.shared.array(shape=cache_shape, dtype=float32)
    q_cache = cuda.shared.array(shape=cache_shape, dtype=float32)
    min_energy_values_cache = cuda.shared.array(shape=(cache_shape, cache_shape), dtype=float32)
    n = p.shape[0]
    i, j = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x

    if i >= n-1 or j >= n-1:
        return

    k = i - width
    l = j - width

    if k < 1:
        k = 1
    if l < 1:
        l = 1
    e = 0.
    minimum = np.inf

    if (tx == 0 and i != 0) and (ty == 0 and j != 0):
        for index in range(width):
            p_cache[index] = p[i - width + index]
        for index in range(width):
            q_cache[index] = q[j - width + index]
    if tx == 0:
        q_cache[ty + width] = q[j]
    if ty == 0:
        p_cache[tx + width] = p[i]
    min_energy_values_cache[tx + width][ty + width] = min_energy_values[i][j]
    if i - width >= 0:
        min_energy_values_cache[tx][ty + width] = min_energy_values[i - width][j]
    if j - width >= 0:
        min_energy_values_cache[tx + width][ty] = min_energy_values[i][j-width]
    if i - width >= 0 and j - width >= 0:
        min_energy_values_cache[tx][ty] = min_energy_values[i-width][j - width]
    l_0 = by * bw - width
    k_0 = bx * bw - width
    cuda.syncthreads()
    if i <= 1 or j <= 1:
        return
    while k < i:
        l = j - width
        if l < 1:
            l = 1
        while l < j:
            e = min_energy_values_cache[k-k_0][l-l_0] + integrate(p_cache, q_cache, k, i, l, j, 1/(n-1), k_0, l_0)
            if e < minimum:
                minimum = e
            l = l + 1
        k = k + 1
    min_energy_values[i][j] = minimum
    if i == n-2 and j == n-2:
        m[0] = minimum

@cuda.jit
def relax_final(p, q, min_energy_values, path_nodes):
    p_cache = cuda.shared.array(shape=cache_shape, dtype=float32)
    q_cache = cuda.shared.array(shape=cache_shape, dtype=float32)
    min_energy_values_cache = cuda.shared.array(shape=(cache_shape, cache_shape), dtype=float32)
    n = p.shape[0]
    i, j = cuda.grid(2)

    if i >= n-1 or j >= n-1:  # Check array boundaries
        return

    k = i - width
    l = j - width

    if k < 1:
        k = 1
    if l < 1:
        l = 1
    e = 0.
    minimum = np.inf
    index1 = 0
    index2 = 0
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    if (tx == 0 and i != 0) and (ty == 0 and j != 0):
        for index in range(width):
            p_cache[index] = p[i - width + index]
        for index in range(width):
            q_cache[index] = q[j - width + index]
    if tx == 0:
        q_cache[ty + width] = q[j]
    if ty == 0:
        p_cache[tx + width] = p[i]
    min_energy_values_cache[tx + width][ty + width] = min_energy_values[i][j]
    if i - width >= 0:
        min_energy_values_cache[tx][ty + width] = min_energy_values[i - width][j]
    if j - width >= 0:
        min_energy_values_cache[tx + width][ty] = min_energy_values[i][j-width]
    if i - width >= 0 and j - width >= 0:
        min_energy_values_cache[tx][ty] = min_energy_values[i-width][j - width]
    l_0 = by * bw - width
    k_0 = bx * bw - width
    cuda.syncthreads()
    if i <= 1 or j <= 1:
        return
    while k < i:
        l = j - width
        if l < 1:
            l = 1
        while l < j:
            e = min_energy_values_cache[k-k_0][l-l_0] + integrate(p_cache, q_cache, k, i, l, j, 1/(n-1), k_0, l_0)
            if e < minimum:
                minimum = e
                index1 = k
                index2 = l
            l = l + 1
        k = k + 1
    min_energy_values[i][j] = minimum
    path_nodes[i][j][0] = index1
    path_nodes[i][j][1] = index2


@jit(nopython=True)
def interp_uniform(t, interval, y, l_0):
    i = int(t/interval) - l_0
    if i == int(1/interval) - l_0:
        return y[i]
    else:

        return (t - interval * (i + l_0)) * (y[i + 1] - y[i]) / interval + y[i]

@jit()
def integrate(p, q, k, i, l, j, gamma_interval, k_0, l_0):
    e = 0
    a = k
    gammak_1 = 0
    gammak_2 = 0
    while a < i:
        gammak_1 = (l + (a-k) * (j - l) / (i-k)) * gamma_interval
        gammak_2 = (l + (a - k + 1) * (j - l) / (i - k)) * gamma_interval
        e = e + (0.5 * (p[a - k_0] - interp_uniform(gammak_1, gamma_interval, q, l_0)) ** 2
                 + 0.5 * (p[(a + 1 - k_0)] - interp_uniform(gammak_2, gamma_interval, q, l_0)) ** 2) * \
                (gamma_interval) * 0.5
        a = a + 1
    return e


# @jit(nopython=True)
# def integrate(p, q, k, i, l, j, gamma_interval, k_0, l_0):
#     a = k
#
#     gammak_1 = l * gamma_interval
#     gammak_2 = (l + (i-k) * (j - l) / (i - k)) * gamma_interval
#     e = (0.5 * (p[k - k_0] - interp_uniform(gammak_1, gamma_interval, q, l_0)) ** 2
#                  + 0.5 * (p[(i - k_0)] - interp_uniform(gammak_2, gamma_interval, q, l_0)) ** 2) * \
#                 (gamma_interval) * 0.5
#     return e

