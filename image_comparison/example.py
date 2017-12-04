from operator import index

import image_comparison.curve_processing
from scipy.io import loadmat
from sqlalchemy.sql.functions import concat
from shapedist import elastic_linear
import numpy as np
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from scipy import stats


def find_nearest(value, array):
    idx = (np.abs(array-value)).argmin()
    if idx < array.size-1:
        idx = idx
    return idx

def task(p, q):
    temp_domain = np.linspace(0., 1., p[0].size)
    temp_interp = interp1d(temp_domain, p[1])
    temp_domain = np.linspace(0., 1., q[0].size)
    temp_p = temp_interp(temp_domain)
    fft_array = np.absolute(np.fft.ifft(np.fft.fft(temp_p)))
    val = fft_array.max()
    index1 = find_nearest(val, p[1])
    index2 = find_nearest(val, q[1])
    temp = [[], []]
    temp[0] = np.array(p[0][index1:-1].tolist() + p[0][:index1-1].tolist() + [p[0][index1:-1].tolist()[0]])
    temp[1] = np.array(p[1][index1:-1].tolist() + p[1][:index1 - 1].tolist() + [p[1][index1:-1].tolist()[0]])
    p = np.array(temp)

    temp = [[], []]
    temp[0] = np.array(q[0][index2:-1].tolist() + q[0][:index2-1].tolist() + [q[0][index2:-1].tolist()[0]])
    temp[1] = np.array(q[1][index2:-1].tolist() + q[1][:index2 - 1].tolist() + [q[1][index2:-1].tolist()[0]])
    q = np.array(temp)

    s1, theta1 = image_comparison.curve_processing.convert_to_angle_function(p[0], p[1])
    s2, theta2 = image_comparison.curve_processing.convert_to_angle_function(q[0], q[1])
    t = np.array(np.union1d(s1, s2))

    angle_1_interpolation = interp1d(s1, theta1)
    angle_2_interpolation = interp1d(s2, theta2)
    theta1 = np.array(angle_1_interpolation(t))
    theta2 = np.array(angle_2_interpolation(t))
    tg, gamma, energy = elastic_linear.find_gamma(np.array((t, theta1)),
                                                  np.array((t, theta2)),
                                                  6, 12, 7)
    gamma_interpolation = interp1d(tg, gamma)
    theta2 = angle_2_interpolation(gamma_interpolation(t))
    energy = image_comparison.curve_processing.integrate(t, theta1, theta2)
    return energy

all_curves = loadmat('Curve_data.mat')
print( all_curves.keys() )
print( all_curves['MPEG7_classes'] )
curves_nonuniform = all_curves['MPEG7_curves_64']
print("hi")
print(task(curves_nonuniform[0][0], curves_nonuniform[0][0]))
