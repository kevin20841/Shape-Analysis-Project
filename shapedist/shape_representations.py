import numpy as np
import shapedist

def coords(p):
    return p, None


def tangent(p):
    tans = p[1:] - p[:-1]
    bot = np.sqrt(tans[:, 0] **2 + tans[:, 1] **2)
    tans = np.divide(tans, bot)
    return tans
    # x = p[:, 0]
    # y = p[:, 1]
    # dx_dt = np.gradient(x)
    # dy_dt = np.gradient(y)
    # velocity = np.array([[dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
    # ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
    # tangent = np.array([1 / ds_dt] * 2).transpose() * velocity
    # return np.arctan2(tangent[:, 0], tangent[:, 1]), None


def normals(p, t):
    x = p[:, 0]
    y = p[:, 1]
    dx_dt = np.gradient(x, t)
    dy_dt = np.gradient(y, t)
    velocity = np.array([[dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
    tangent = np.array([1 / ds_dt] * 2).transpose() * velocity
    tangent_x = tangent[:, 0]
    tangent_y = tangent[:, 1]

    deriv_tangent_x = np.gradient(tangent_x)
    deriv_tangent_y = np.gradient(tangent_y)

    dT_dt = np.array([[deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size)])

    length_dT_dt = np.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)
    for i in range(length_dT_dt.shape[0]):
        if length_dT_dt[i] == 0:
            length_dT_dt[i] = 1

    normal = np.array([1 / length_dT_dt] * 2).transpose() * dT_dt

    for i in range(len(normal)):
        if np.cross(tangent[i], normal[i]) > 0:
            normal[i] = -1 * normal[i]
    return normal, None


def curvature(p):
    return shapedist.build_hierarchy.curvature(p), None


def angle_function(p):
    x = p[:, 0]
    y = p[:, 1]
    i = 1
    com_x = 0
    com_y = 0
    curve_length = 0
    s = np.zeros(x.size-1, dtype=np.float64)

    while i < x.size - 1:
        h_i = ((x[i] - x[i+1])**2 + (y[i] - y[i+1])**2)**0.5
        h_i_1 = ((x[i+1] - x[i]) ** 2 + (y[i+1] - y[i]) ** 2) ** 0.5
        com_x = com_x + (h_i + h_i_1) / 2 * x[i]
        com_y = com_y + (h_i + h_i_1) / 2 * y[i]
        curve_length = curve_length + h_i_1
        s[i] = curve_length
        i = i + 1
    curve_length = curve_length + ((x[x.size - 1] - x[0]) ** 2 + (y[y.size - 1] - y[0]) ** 2) ** 0.5
    x = (x - com_x) / curve_length
    y = (y - com_y) / curve_length
    s = s / curve_length

    theta_0 = np.zeros(s.size, dtype=np.float64)
    i = 1
    while i < theta_0.size:
        theta_0[i] = np.arctan2(y[i+1], x[i+1])
        i = i + 1
    integral = 0
    while i < s.size-1:
        integral = integral + (theta_0[i] + theta_0[i+1]) / 2 * (s[i+1] - s[i])
        i = i + 1
    C = np.pi - integral
    theta = theta_0 + C

    i = 0
    while i < theta.size - 1:
        angle_change = theta[i+1] - theta[i]
        if np.fabs(angle_change) > np.pi:
            theta[i + 1] = theta[i + 1] - 2 * np.pi * np.sign(angle_change)
            pass
        i = i + 1
    theta = np.abs(theta)
    # theta = theta / (2 * np.pi) + 0.5

    return s, theta


def srvf(p, t):
    x = p[:, 0]
    y = p[:, 1]
    dx_dt = np.gradient(x, t)
    dy_dt = np.gradient(y, t)
    result = np.zeros((x.size, 2), dtype=np.float64)
    mag = (dx_dt ** 2 + dy_dt ** 2) ** 0.25

    result[:, 0] = dx_dt / mag
    result[:, 1] = dy_dt / mag

    return result, None


def calculate_com(p):
    com = np.zeros(p.shape[1])
    i = 1
    while i < p.shape[1] - 1:
        h1 = np.sum(np.power(p[i] - p[i-1], 2))**0.5
        h2 = np.sum(np.power(p[i+1] - p[i], 2))**0.5
        com = com + (h1 + h2) / 5 *p[i]
        i = i + 1
    return com
