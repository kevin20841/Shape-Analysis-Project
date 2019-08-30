import numpy as np
import os
import sys
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "../")
sys.path.append(filename)
import shapedist
from shapedist import normalize

from skimage import measure
from skimage import io
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import time


def smooth_curve(b, iter):
    b = normalize(b)
    for k in range(iter):
        N = b.shape[0]
        b_f = rotate(b, 1)
        b_b = rotate(b, N-1)
        d = np.linalg.norm(b_f - b, axis=1)
        d_b = rotate(d, N-1)
        smoothed = 0.5 * b +0.5 * div((mult(d_b, b_f) + mult(d, b_b)), d_b + d)
        b = np.copy(smoothed)
    return smoothed

def div(b, d):
    N = d.shape[0]
    ret = np.zeros(b.shape)
    for i in range(N):
        ret[i] = b[i] / d[i]
    return ret

def mult(d, b):
    N = d.shape[0]
    ret = np.zeros(b.shape)
    for i in range(N):
        ret[i] = d[i] * b[i]
    return ret
def rotate(p, s):
    ret = np.copy(p)
    N = p.shape[0]
    temp = ret[N - s:]
    ret[s:N] = ret[0:N - s]
    ret[:s] = temp
    return ret


dirname = os.path.dirname(__file__)
dir = "./img/BBC020_v1_outlines_cells"
uncoarsened  = []
coarsened = []
images = os.listdir(dir)
unsizes = []
coarsesizes = []
percents = []
times = []
uniform_256 = []
for name in tqdm(images):
    img = io.imread(dir + "/" + name)
    mini = np.inf
    contours = measure.find_contours(img, level=0.8)
    max = 0
    index = 0
    for i in range(len(contours)):
        if contours[i].shape[0] > max:
            index = i
            max = contours[i].shape[0]
    if max < 100:
        continue

    sm = smooth_curve(contours[index], iter=10)
    t = shapedist.arclen_fct_values(sm)
    func1 = CubicSpline(t, sm[:, 0])
    func2 = CubicSpline(t, sm[:, 1])
    t_new = np.linspace(0., 1., 1024)
    sm_uniform = np.zeros((1024, 2))
    sm_uniform[:, 0] = func1(t_new)
    sm_uniform[:, 1] = func2(t_new)


    uncoarsened.append(sm_uniform)
    t_256 = np.linspace(0., 1., 256)
    sm_256 = np.zeros((256, 2))
    sm_256[:, 0] = func1(t_256)
    sm_256[:, 1] = func2(t_256)
    uniform_256.append(sm_256)

    temp, coarse = shapedist.build_hierarchy.coarsen_curve(t_new, sm_uniform, tol=1e-4, maxiter=15)
    unsizes.append(sm.shape[0])
    coarsesizes.append(coarse.shape[0])
    coarsened.append(coarse)
    percents.append(1 - coarse.shape[0] / sm_uniform.shape[0])

    for i in range(5):
        start = time.time()
        contours = measure.find_contours(img, level=0.5)
        sm = smooth_curve(contours[0], iter=10)

        t = np.arange(sm_uniform.shape[0])
        temp, coarse = shapedist.build_hierarchy.coarsen_curve(t, sm_uniform, tol=1e-4, maxiter=15)
        end = time.time()

        if end - start < mini:
            mini = end - start
    times.append(mini)

#
uniform_256 = np.array(uniform_256)
uncoarsened = np.array(uncoarsened)
coarsened = np.asarray(coarsened)
unsizes = np.array(unsizes)
coarsesizes = np.array(coarsesizes)
percents = np.array(percents)
times = np.array(times)

np.save("marrow_cell_curves_256", uniform_256)
np.save("marrow_cell_curves_1024", uncoarsened)
np.save("marrow_cell_curves_coarsened", coarsened)
np.savetxt("marrow_cell_curves_full_sizes", unsizes)
np.savetxt("marrow_cell_curves_coarsened_sizes", coarsesizes)
np.savetxt("marrow_cell_curves_percent_red", percents)
np.savetxt("marrow_cell_curves_process_times", times)

print("Num Curves:", uniform_256.shape[0])
print("Average full size:", np.sum(unsizes) / unsizes.shape[0])
print("Average coarsened size (with tol 2e-3):", np.sum(coarsesizes) / coarsesizes.shape[0])
print("Average percent reduction:", np.sum(percents) / percents.shape[0])
print("Average Time", np.sum(times) / times.shape[0])