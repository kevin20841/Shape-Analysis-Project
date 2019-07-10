import numpy as np


def calculate_tangent(x, y):
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    velocity = np.array([[dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
    tangent = np.array([1 / ds_dt] * 2).transpose() * velocity
    return tangent


def calculate_normals(p):
    x = p[:, 0]
    y = p[:, 1]
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    velocity = np.array([[dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
    tangent = np.array([1 / ds_dt] * 2).transpose() * velocity
    tangent_x = tangent[:, 0]
    tangent_y = tangent[:, 1]

    deriv_tangent_x = np.gradient(tangent_x)
    deriv_tangent_y = np.gradient(tangent_y)

    dT_dt = np.array([[deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size)])

    length_dT_dt = np.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)

    normal = np.array([1 / length_dT_dt] * 2).transpose() * dT_dt
    for i in range(len(normal)):
        if np.cross(tangent[i], normal[i]) > 0:
            normal[i] = -1 * normal[i]
    return normal


def calculate_curvature(x, y, closed):
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5

    return curvature


def calculate_angle_function(x, y):
    i = 1
    h_i = 0
    h_i_1 = 0
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
    theta = theta / (2 * np.pi) + 0.5

    return s, theta


def calculate_srvf(x, y):
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    return dx_dt / np.sqrt(np.linalg.norm(dx_dt)), dy_dt / np.sqrt(np.linalg.norm(dy_dt))