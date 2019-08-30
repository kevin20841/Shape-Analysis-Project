import numpy as np
from numba import njit

@njit(cache=True)
def nsum(arr):
    s = 0
    for i in arr:
        s = s + i
    return i

@profile
def main():
    arr = np.linspace(0., 1., 10000000)
    nsum(arr)
    nsum(arr)
    np.sum(arr)
    print("Hello World!")

if __name__ == "__main__":
    main()
