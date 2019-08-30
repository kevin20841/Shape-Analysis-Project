import numpy as np
import shapedist
import scipy
from scipy.integrate import trapz
from scipy.linalg import svd
import scipy.interpolate
from numba import jit, float64

def curvature_shapedist(p, q, dr='', neigh=5, distfunc=None, t1=None, t2=None,
                   tol=2e-3, energy_dot=False):
    shape_rep = shapedist.curvature
    p = np.copy(p)
    q = np.copy(q)
    N = p.shape[0]
    m = np.inf
    dr = dr +  "d"
    dist = 0
    if 'u' in dr.lower():
        t1 = np.linspace(0., 1., N)
        t2= t1
    s = N // 100
    for i in range(100):
        sdist, p_temp, q_temp, t, gamma = shapedist.find_shapedist(p, q, dr, shape_rep=shape_rep, distfunc=distfunc, t1=t1, t2=t2,
                   tol=tol, energy_dot=energy_dot, neigh=neigh)
        if sdist < m:
            m = sdist
        temp = p[N-s:]
        p[s:N] = p[0:N-s]
        p[:s] = temp
    return m


def closed_curve_shapedist(p, q, dr='', neigh=5, shape_rep=shapedist.coords, distfunc=None, t1=None, t2=None,
                   tol=2e-3, energy_dot=False):
    p = np.copy(p)
    q = np.copy(q)
    N = p.shape[0]
    m = np.inf
    dr = dr +  "d"
    dist = 0
    if 'u' in dr.lower():
        t1 = np.linspace(0., 1., N)
        t2= t1
    s = N // 25
    if shape_rep is shapedist.srvf:
        energy_dot = True
        distfunc = shapedist.calculate_shape_distance_SRVF
    for i in range(25):


        sdist, p_temp, q_temp, t, gamma = shapedist.find_shapedist(p, q, dr, shape_rep=shape_rep, distfunc=distfunc, t1=t1, t2=t2,
                   tol=tol, energy_dot=energy_dot, neigh=neigh)
        print("1.", sdist )
        q_reparam = np.zeros(q_temp.shape)
        gammad = np.sqrt(np.gradient(gamma, t))
        for j in range(q.shape[1]):
            func = scipy.interpolate.CubicSpline(t, q_temp[:, j])
            q_reparam[:, j] = func(gamma)
            if shape_rep is shapedist.srvf:
                q_reparam[:, j] = np.multiply(q_reparam[:, j], gammad)
        R = optimal_rotation(p_temp, q_reparam, t)
        for j in range(p_temp.shape[0]):
            p_temp[j] =  R @ p_temp[j]
        sdist, p_temp, q_temp, t, gamma = shapedist.find_shapedist(p_temp, q_temp, dr, shape_rep=shapedist.coords, distfunc=distfunc,
                                                         t1=t1, t2=t2, tol=tol, energy_dot=energy_dot, neigh=neigh)
        if sdist < m:
            m = sdist
        temp = p[N - s:]
        p[s:N] = p[0:N-s]
        p[:s] = temp

    return m

def closed_curve_tangent_shapedist(p, q, dr='', shape_rep=shapedist.coords, distfunc=None, t1=None, t2=None,
                   init_coarsening_tol=2e-4, energy_dot=False):
    p = np.copy(p)
    q = np.copy(q)
    N = p.shape[0]
    m = np.inf
    dr = dr +  "d"
    if 'u' in dr.lower():
        t1 = np.linspace(0., 1., N)
        t2= t1
    s = N // 50
    tanp, temp = shapedist.tangent(p)
    averagep = trapz(tanp) / tanp.shape[0]

    for i in range(50):
        sdist, p_temp, q_temp, t, gamma = shapedist.find_shapedist(p, q, dr, shape_rep, distfunc, t1, t2,
                   init_coarsening_tol, energy_dot)
        func = scipy.interpolate.CubicSpline(t, q_temp)
        q_reparam= func(gamma)

        averageq = trapz(q_reparam) /q_reparam.shape[0]
        theta_0 = averageq-averagep
        p_temp = p_temp + theta_0
        p_temp = p_temp % (2 * np.pi)
        sdist, p_temp, q_temp, t, gamma = shapedist.find_shapedist(p_temp, q_temp, dr, shapedist.coords, distfunc, t1, t2,
                   init_coarsening_tol, energy_dot)
        if sdist < m:
            m = sdist
        temp = p[N-s:]
        p[s:N] = p[0:N-s]
        p[:s] = temp

    return m






def optimal_rotation(p, q, t=None):
    (N, dim) = p.shape
    A = np.zeros((dim, dim))
    if t is None:
        h  =1 / (N-1)
        for i in range(dim):
            for j in range(dim):
                A[i][j] = h * trapz(np.multiply(p [:, i], q[:, j]))
    else:
        for i in range(dim):
            for j in range(dim):
                A[i][j] = trapz(t,np.multiply(p [:, i], q[:, j]))

    U, temp, V = svd(A)
    if np.linalg.det(A) > 0:
        S = np.identity(dim)
    else:
        S = np.identity(dim)
        S[-1, -1] = -S[-1, -1]
    return U@S@V