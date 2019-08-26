import sys
import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "../")
sys.path.append(filename)
from joblib import Parallel, delayed, dump, load
import shapedist
import numpy as np


def job(start, end):
    # load data from disk
    folder = os.path.join(dirname, "./jm")
    dfm = os.path.join(folder, 'data_memmap')
    curves = load(dfm, mmap_mode='r')


    matrix = Parallel(n_jobs=-1, verbose=30, max_nbytes='50M')(delayed(task)(curves[i//curves.shape[0]], curves[i%curves.shape[0]]) for i in range(start *curves.shape[0], end * curves.shape[0]))
    matrix = np.reshape(np.array(matrix), (end-start, curves.shape[0]))

    np.savetxt(os.path.join(dirname, "./output/" + str(start) +"_"+str(end)), matrix)

def task(p, q):
    # t = np.linspace(0., 1., p.shape[0])
    # dist = shapedist.closed_curve_shapedist(p, q, dr='', shape_rep=shapedist.coords)

    # kd_dist = 1e-14 if kd_dist < 1e-14 else kd_dist
    kd_dist, temp = shapedist.kd_tree.shape_dist(p, q)
    return kd_dist

def main():
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    job(start, end)

if __name__ == "__main__":
    sys.exit(main())
