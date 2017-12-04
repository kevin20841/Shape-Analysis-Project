import numpy as np
from numba import jit, types, float64, int16
import shapedist.elastic_n_2
from math import floor

@jit(int16(float64, float64), nopython=True)
def test(x, y):
    return x // y

print(test(5.6, 2.8))