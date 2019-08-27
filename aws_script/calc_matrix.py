import sys
import os
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "../")
sys.path.append(filename)
from joblib import Parallel, delayed, dump, load
import shapedist
import numpy as np

filename = "cell_curves_coarsened_multi_srvf"
def cache():
    # put data onto disk for easy access
    folder = "./jm"
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    dfm = os.path.join(folder, 'data_memmap')
    from scipy.io import loadmat

    # all_curves = loadmat('../data/Curve_data.mat')
    # curves_raw = all_curves['MPEG7_curves_256']
    curves = np.load("../data/marrow_cell_curves_256.npy", allow_pickle=True)
    curves = curves[:200]
    # for i in range(100):
    #     curves[i] = curves_raw[i][0].T
    # for i in range(10):
    #     for j in range(2):
    #         curves[i * 2 + j] = curves_raw[i * 10 + j][0].T
    dump(curves, dfm)

def job():
    # load data from disk
    folder = os.path.join(dirname, "./jm")
    dfm = os.path.join(folder, 'data_memmap')
    curves = load(dfm, mmap_mode='r')


    matrix = Parallel(n_jobs=-1, verbose=8, max_nbytes='50M')(delayed(task)(curves[i//curves.shape[0]], curves[i%curves.shape[0]])
                                                               for i in range(0, curves.shape[0] * curves.shape[0]))
    matrix = np.reshape(np.array(matrix), (curves.shape[0], curves.shape[0]))

    np.save(os.path.join(dirname, "./output/" +filename), matrix)

def task(p, q):
    t = np.linspace(0., 1., p.shape[0])
    # dist = shapedist.closed_curve_shapedist(p, q, t1=t, t2=t, dr='', shape_rep=shapedist.srvf)
    dist = shapedist.find_shapedist(p, q, t1=t, t2=t, dr='', shape_rep=shapedist.coords)
    return dist
    # kd_dist, temp = shapedist.kd_tree.shape_dist(p, q)
    # return kd_dist
def main():
    args = sys.argv
    if len(args) == 1:
        print("Caching")
        cache()
        print("Computing")
        job()
        print("Finished!")
    if "-c" in args:
        cache()
    if "-j" in args:
        job()

if __name__ == "__main__":
    sys.exit(main())
