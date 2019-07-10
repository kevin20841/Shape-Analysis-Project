from numba import cuda, float32, jit, SmartArray
import numpy as np
np.set_printoptions(threshold=np.nan, linewidth= 400)
width = 4


def elastic_gpu_optimized_1D(p, q):
    n = p.size
    # Load data
    min_energy_values = np.full((n, n), np.inf, dtype=np.float32)
    path_nodes = np.zeros((n, n, 2), dtype=np.int16)
    min_energy_values[1][1] = integrate(p, q, 0, 1, 0, 1, 1/(n-1))
    p = SmartArray(p)
    q = SmartArray(q)
    min_energy_values = SmartArray(min_energy_values)
    path_nodes = SmartArray(path_nodes)
    # Set the number of threads in a block
    threadsperblock = 128

    # Calculate the number of thread blocks in the grid
    # blockspergrid = (1, 1)
    # relax_edges
    m = SmartArray(np.full(1, 3, dtype=np.int16))
    for i in range(2 * n - 2):
        blockspergrid = int((m[0] + (threadsperblock - 1)) // threadsperblock)
        relax[blockspergrid, threadsperblock](p, q, min_energy_values, path_nodes, m)
        print(i)
        # if prev != np.inf and prev == min_energy_values[-2][-2]:
        #     break
        # prev = min_energy_values[-2][-2]
    # Print the result
    gamma_interval = 1/(n-1)

    i = n - 1
    j = n - 1
    min_energy_values[i][j] = integrate(p, q, 0, i, 0, j, gamma_interval)
    k = i - width
    if k <= 0:
        k = 0
    minimum = min_energy_values[i][j]
    while k < i:
        l = j - width
        if l <= 0:
            l = 0
        while l < j:
            e = min_energy_values[k, l] + integrate(p, q, k, i, l, j, gamma_interval)
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
def relax(p, q, min_energy_values, path_nodes, m):
    n = p.shape[0]
    pos = cuda.grid(1)
    i = pos
    j = m[0] - pos
    temp = m[0]
    if pos == 0:
        m[0] = m[0] + 1
    if i >= n-1 or j >= n-1:  # Check array boundaries
        return
    if i <= 0 or j <= 0:
        return
    if i == 1 or j == 1:
        min_energy_values[i][j] = integrate(p, q, 0, i, 0, j, 1/(n-1))
        return
    if i + j != temp:
        return
    k = i - width
    l = j - width

    if k < 1:
        k = 1

    e = 0.

    minimum = np.inf
    final_index_1 = 0
    final_index_2 = 0
    while k < i:
        l = j - width
        if l < 1:
            l = 1
        while l < j:
            e = min_energy_values[k][l] + integrate(p, q, k, i, l, j, 1/(n-1))
            if e < minimum:
                minimum = e
                final_index_1 = k
                final_index_2 = l
            l = l + 1
        k = k + 1
    min_energy_values[i][j] = minimum
    path_nodes[i][j][0] = final_index_1
    path_nodes[i][j][1] = final_index_2
    cuda.syncthreads()




@jit()
def interp_uniform(t, interval, y):
    i = int(t / interval)
    if i == int(1/interval):
        return y[i]
    else:

        return (t - interval * i) * (y[i + 1] - y[i]) / interval + y[i]

@jit()
def integrate(p, q, k, i, l, j, gamma_interval):
    e = 0
    a = k
    gammak_1 = 0
    gammak_2 = 0
    while a < i:
        gammak_1 = (l + (a-k) * (j - l) / (i-k)) * gamma_interval
        gammak_2 = (l + (a - k + 1) * (j - l) / (i - k)) * gamma_interval
        e = e + (0.5 * (p[a] - interp_uniform(gammak_1, gamma_interval, q)) ** 2
                 + 0.5 * (p[(a + 1)] - interp_uniform(gammak_2, gamma_interval, q)) ** 2) * \
                (gamma_interval) * 0.5
        a = a + 1
    return e
