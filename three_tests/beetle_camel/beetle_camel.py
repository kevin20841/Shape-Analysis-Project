import sys
sys.path.append("../../")
sys.path.append("../../data")
import matplotlib.pyplot as plt
import shapedist
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import scipy.interpolate
import time
def figure_plot(p, q, tg, gamma, shape_rep, name, alg):

    t = tg
    if shape_rep is shapedist.coords:
        plt.figure()
        plt.plot(p[:, 0], p[:, 1], ".-b")
        plt.plot(q[:, 0], q[:, 1], ".-g")
        q_reparam = np.zeros(q.shape)
        for d in range(p.shape[1]):
            q_reparam[:, d] = CubicSpline(t, q[:, d])(gamma)
        plt.plot(q_reparam[:, 0], q_reparam[:, 1], ".r")

    elif shape_rep is shapedist.tangent:
        plt.figure()
        q_reparam = np.zeros(q.shape)
        for d in range(p.shape[1]):
            q_reparam[:, d] = CubicSpline(t, q[:, d])(gamma)

        pt = np.arctan2(p[:, 1], p[:, 0])
        qt = np.arctan2(q_reparam[:, 1], q_reparam[:, 0])
        plt.plot(t, pt, "-b")
        plt.plot(tg, qt, "-r")
        plt.plot(t,  np.arctan2(q[:, 1], q[:, 0]), "-g")
    elif shape_rep is shapedist.srvf:
        plt.figure()
        q_reparam = np.zeros(q.shape)
        gammad = np.sqrt(np.gradient(gamma, t))
        for j in range(q.shape[1]):
            func = scipy.interpolate.CubicSpline(t, q[:, j])
            q_reparam[:, j] = func(gamma)
            q_reparam[:, j] = np.multiply(q_reparam[:, j], gammad)
        plt.subplot(2, 1, 1)
        plt.plot(t, p[:, 0], "-b")
        plt.plot(t, q[:, 0], "-g")
        plt.plot(tg, q_reparam[:, 0], "r")

        plt.subplot(2, 1, 2)
        plt.plot(t, p[:, 1], "-b")
        plt.plot(t, q[:, 1], "-g")
        plt.plot(tg, q_reparam[:, 1], "r")
    else:
        return
    plt.savefig("./img/" + name + "/" + name + "_" + alg + "_" + shape_rep.__name__.lower() + ".png")
    plt.close()

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    print(*args)

def run_tests():
    all_curves = loadmat('../../data/Curve_data.mat')

    curves= all_curves['MPEG7_curves_256']
    bat_1 = [curves[20][0].T[:-1]]
    p = curves[70][0].T[:-1]
    s = p.shape[0] - np.argmax(p[:, 0])
    bat_2 = [np.roll(p, s, axis=0)]


    curves= all_curves['MPEG7_curves_1024']
    bat_1.append(curves[20][0].T[:-1])
    p = curves[70][0].T[:-1]
    s = p.shape[0] - np.argmax(p[:, 0])
    bat_2.append(np.roll(p, s, axis=0))
    #
    curves= all_curves['MPEG7_curves_coarsened']
    bat_1.append(curves[20][0].T[:-1])
    p = curves[70][0].T[:-1]
    s = p.shape[0] - np.argmax(p[:, 0])
    bat_2.append(np.roll(p, s, axis=0))

    shape_reps = [shapedist.coords, shapedist.tangent,shapedist.srvf]
    names = ["256", "1024", "coarsened"]
    # names = ["256", "1024"]
    # tolerance: 0.02
    for i in range(len(bat_1)):
        eprint("Discretization: " + names[i])
        eprint("-" * 162)
        eprint("{: <15}| {: <15}| {: <15}| {: <15}| {: <15}".format("Shape rep:", "N_2", "Uni-Multi",
                                                           "N_2 t", "Uni-Multi t"))
        for shape_rep in shape_reps:
            t = np.linspace(0., 1., bat_1[i].shape[0]) if names[i] != "coarsened" else None
            uarg = "u2" if names[i] != "coarsened" else "2"
            marg = "um" if names[i] != "coarsened" else "m"
            renergy, p3, q3, tg3, gamma3 = shapedist.find_shapedist(bat_1[i], bat_2[i], '' + 'd', t1=t, t2=t,
                                                                    shape_rep=shape_rep)
            uenergy, p1, q1, tg1, gamma1 = shapedist.find_shapedist(bat_1[i], bat_2[i], uarg + 'd', t1=t, t2=t,
                                                                    shape_rep=shape_rep)
            time1 = time.time()
            uenergy, p1, q1, tg1, gamma1 = shapedist.find_shapedist(bat_1[i], bat_2[i], uarg + 'd', t1=t, t2=t, shape_rep=shape_rep)
            time1 = time.time() - time1
            figure_plot(p1, q1, tg1, gamma1, shape_rep, names[i], "Uniform_N2")

            time2 = time.time()
            menergy, p2, q2, tg2, gamma2 = shapedist.find_shapedist(bat_1[i], bat_2[i], marg + 'd', t1=t, t2=t, shape_rep=shape_rep)
            time2 = time.time()- time2
            figure_plot(p2, q2, tg2, gamma2, shape_rep, names[i], "Uniform_Multi")

            plt.figure()
            plt.plot(tg1, gamma1, "-g")
            plt.plot(tg2, gamma2, ".b")
            plt.plot(tg3, gamma3, ".r")
            plt.savefig("./img/" + names[i] + "/" + names[i] + "_" + shape_rep.__name__.lower()+"_gamma.png")
            plt.close()

            eprint("{: <15}| {: <15}| {: <15}| {: <15}| {: <15}".format(shape_rep.__name__.lower(), "%0.10f" % uenergy,
                                                                                                  "%0.10f" % menergy,
                                                                                                  "%0.10f"%time1, "%0.10f"%time2,
                                                                                         ))
        eprint()
        eprint()

def main():
    run_tests()

if __name__ == "__main__":
    sys.exit(main())
