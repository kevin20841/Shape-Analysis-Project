import numpy as np
import matplotlib.pyplot as plt
from numba import jit, types, float64, int16
from scipy.io import loadmat
from sqlalchemy.sql.functions import concat
np.set_printoptions(threshold=np.nan, linewidth= 400)


@jit([types.Tuple((float64[:], float64[:]))(float64[:], float64[:])], cache=True, nopython=True)
def convert_to_angle_function(x, y):
    i = 1
    h_i = 0
    h_i_1 = 0
    com_x = 0
    com_y = 0
    curve_length = 0
    s = np.zeros(x.size, dtype=np.float64)
    while i < x.size - 1:
        h_i = ((x[i] - x[i+1])**2 + (y[i] - y[i+1])**2)**0.5
        h_i_1 = ((x[i-1] - x[i]) ** 2 + (y[i-1] - y[i]) ** 2) ** 0.5
        com_x = com_x + (h_i + h_i_1) / 2 * x[i]
        com_y = com_y + (h_i + h_i_1) / 2 * y[i]
        curve_length = curve_length + h_i_1
        s[i] = curve_length
        i = i + 1
    h_i_1 = ((x[i-1] - x[i]) ** 2 + (y[i-1] - y[i]) ** 2) ** 0.5
    curve_length = curve_length + h_i_1
    x = (x - com_x) / curve_length
    y = (y - com_y) / curve_length
    s = s / curve_length
    s[s.size-1] = 1
    theta_0 = np.zeros(x.size, dtype=np.float64)
    i = 0
    while i < theta_0.size:
        theta_0[i] = np.arctan2(y[i], x[i])
        i = i + 1
    i = 0
    integral = 0
    while i < s.size-1:
        integral = integral + (theta_0[i] + theta_0[i+1]) / 2 * (s[i+1] - s[i])
        i = i + 1

    C = np.pi - integral
    theta = theta_0 + C
    return s, theta


@jit([float64(float64[:], float64[:], float64[:])], cache=True, nopython=True)
def integrate(t, theta1, theta2):
    s = 0
    i = 0
    while i < theta1.size - 1:
        s = s + ((theta1[i+1]-theta2[i+1]) ** 2 + (theta1[i]-theta2[i])**2) / 2 * (t[i+1] - t[i])
        i = i + 1
    return s