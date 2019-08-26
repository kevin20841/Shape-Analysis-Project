import numpy as np
import shapedist
from numba import njit, jitclass
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from math import floor

#@njit
def ad_hoc(ind, p, q, t, tol=0.05):
    # N = ind.shape[0]
    # # make all segments monotonic increasing
    #
    # for i in range(0, N-1):
    #     if ind[i] > ind[i+1] and np.abs(ind[i+1] - ind[i])/N < tol:
    #         ind[i+1] = ind[i]
    #
    # for i in range(N):
    #     if np.abs(ind[i] - ind[i+1])/N > tol:
    #         e = i + 1
    #         break
    # ind[0:e] = ind[0:e] - ind[0]
    # i = 1
    # num_segments = 0
    # while i < N:
    #     if np.abs(ind[i] - ind[i-1])/N > tol:
    #         while i < N-1:
    #             if np.abs(ind[i + 1] - ind[i])/N > tol:
    #                 break
    #             i = i + 1
    #         num_segments += 1
    #     i = i + 1
    # i = 1
    # count =1
    # step = floor(N / (num_segments + 1))
    # while i < N:
    #     if np.abs(ind[i] - ind[i-1])/N > tol:
    #         start = i
    #         while i < N-1:
    #             if np.abs(ind[i + 1] - ind[i])/N > tol:
    #                 end = i + 1
    #                 break
    #             i = i + 1
    #         # ind[start:end]  = ind[start:end]- (ind[start] - ind[start-1])
    #         ind[start:end] = ind[start:end] - (ind[start] - step * count)
    #     i = i + 1
    # m = ind[-1]
    # for i in range(N):
    #     ind[i] = floor(ind[i] / m * (N-1))
    return ind

def smooth(x, N, tol = 0.01):
    ret = np.zeros(x.size, dtype= np.int64)
    for i in range(ret.size):
        back = i - N if (i - N) >= 0 else 0
        front = i + N if (i + N) < ret.size else ret.size - 1
        ret[i] = floor(np.sum(x[back:front]) / (front - back))
        if np.abs(ret[i] - x[back]) > tol:
            ret[i] = ret[back]
    return ret
@njit
def flat_dist(p, q, ind, t, start, end):
    s = 0
    arc = t
    for i in range(start, end-1):
        # trapezoidal method
        s = s + 0.5 * (0.5 * norm(p[i], q[ind]) + 0.5 * norm(p[i + 1], q[ind])) * (arc[i + 1] - arc[i])
    return s
@njit
def compute_dist(p, q, ind, t, start, end):
    s = 0
    arc = t
    for i in range(start, end-1):
        # trapezoidal method
        s = s + 0.5 * (0.5 * norm(p[i], q[ind[i]]) + 0.5 * norm(p[i + 1], q[ind[i + 1]])) * (arc[i + 1] - arc[i])
    return s

def get_partitions(ind, tol = 0.05):
    N = ind.shape[0]
    part = np.zeros(ind.shape, dtype = np.bool)
    i = 0
    decreasing=False
    while i < N-1:
        if ind[i] > ind[i + 1] and not decreasing:
            part[i] = True
            decreasing = True
        if ind[i] < ind[i+1] and decreasing:
            decreasing = False
            part[i] = True
        if  abs(ind[i] - ind[i+1])/N > tol:
            part[i] = True
        i = i + 1
    endpoints = []
    for i in range(1, N):
        if part[i]==True and part[i-1] != True:
            endpoints.append(i)

    f = False
    for i in range(endpoints[0] - 1):
        if ind[i] > ind[i+1]:
            f = True
    if f:
        endpoints = [1] + endpoints
    return endpoints

def rebalance(ind):
    N = ind.shape[0]
    while True:
        diff = 0
        t = 0
        for i in range(N-1):
            if ind[i] - ind[i+1] > diff:
                diff = ind[i] - ind[i+1]
                t = i+1
        if diff < ind[-1] - ind[0]:
            break
        ind = np.append(ind[t:], ind[:t])
    return ind


def shape_dist(p, q):
    # normalize

    N = p.shape[0]
    arclen_1 = np.sum((p[1:N, :] - p[0:N - 1, :]) ** 2, 1) ** 0.5
    arclen_1 = np.sum(arclen_1)
    p = (p - shapedist.shape_representations.calculate_com(p)) / arclen_1

    N = q.shape[0]
    arclen_2 = np.sum(np.sum((q[1:N, :] - q[0:N - 1, :]) ** 2, 1) ** 0.5)
    arclen_2 = np.sum(arclen_2)
    q = (q - shapedist.shape_representations.calculate_com(q)) / arclen_2

    t1 = shapedist.arclen_fct_values(p)
    t2 = shapedist.arclen_fct_values(q)

    [t, p, q], mask = shapedist.build_hierarchy.hierarchical_curve_discretization(p, q,
                                                                                  t1, t2,
                                                                                  2e-4,
                                                                                  True, False)

    tree = KDTree(q, leaf_size=2)
    dist, ind = tree.query(p, k=1)
    ind = np.reshape(ind, (-1))
    #
    # ind = rebalance(ind)
    # ind = smooth(ind, 1)
    # # partitions = get_partitions(ind)
    #
    # ind = ad_hoc(ind, p, q, t)
    return compute_dist(p, q, ind, t, 0, p.shape[0]), ind    # (how should we define shape dist?)


@njit
def norm(x, y):
    s = 0
    for i in range(x.shape[0]):
        s = s + (x[i] - y[i])**2
    return s


@njit
def shape_dist_naive(p, q):
    dist = np.empty(q.shape[0])
    ind = np.zeros(q.shape[0], dtype=np.int64)
    for i in range(p.shape[0]):
        m = np.inf
        for j in range(i, i + 30):
            if j >= p.shape[0]:
                break
            d = norm(p[i], q[j])
            if m > d:
                m = d
                index = j
        dist[i] = m
        ind[i] = index
    return np.sum(dist), ind


def distance_matrix(curves):
    numcurves = curves.shape[0]
    distance_matrix = np.empty((numcurves, numcurves))
    for i in range(numcurves):
        tree = KDTree(curves[i], leaf_size=2)
        for j in range(i + 1, numcurves):
            dist, ind = tree.query(curves[j], k=1)
            s = np.sum(dist)
            distance_matrix[i][j] = s
            distance_matrix[j][i] = s
    return distance_matrix