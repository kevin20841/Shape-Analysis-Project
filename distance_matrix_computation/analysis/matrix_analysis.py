import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import itertools
# TODO USE pyxDamerauLevenshtein!!

def kendallTau(A, B):
    pairs = itertools.combinations(range(0, len(A)), 2)

    distance = 0

    for x, y in pairs:
        a = A[x] - A[y]
        b = B[x] - B[y]

        # if discordant (different signs)
        if (a * b < 0):
            distance += 1

    return distance


def compare(n, b):

    bf = b.flatten()
    nf = n.flatten()
    dperc = nf - bf


    for i in range(dperc.shape[0]):
        dperc[i] = 0 if bf[i] == 0 or nf[i] == 0 or bf[i] < 0.05 else (nf[i]-bf[i])/ bf[i]
    m = np.argmax(dperc)
    i = m//515
    j = m - m//515 * 515
    print(m, m//515, m - m//515 * 515)
    print(n[i][j], b[i][j] )
    plt.figure()
    plt.scatter(np.arange(dperc.shape[0]), dperc, s=0.05, marker="o", c = "maroon")
    # print(dperc[dperc > 0.5].shape[0] / dperc.shape[0])
    # dist = []
    # for i in trange(b.shape[0]):
    #     bs = np.asarray(np.argsort(b[i]))
    #     ns = np.asarray(np.argsort(n[i]))
    #     dist.append(kendallTau(bs, ns) / (515 * 515))
    # dist = np.array(dist)
    # print(np.sum(dist) / 515)

DPEM = np.loadtxt("../cell_curves_uniformN_2_srvf_noR.txt")

CEM = np.loadtxt("../cell_curves_cem_srvf_noR.txt")

CCEM2e3 = np.load("../cell_curves_ccem_2e3_srvf_noR.npy")



plt.figure(); plt.imshow(DPEM, cmap='hot', interpolation ="nearest")
plt.figure(); plt.imshow(CEM, cmap='hot', interpolation = "nearest")
plt.figure(); plt.imshow(CCEM2e3, cmap='hot', interpolation ="nearest")

#
compare(CCEM2e3, CEM)
# compare(CEM, DPEM)
# compare(CEM, DPEM)
plt.show()
#compare(CCEM, DPEM)
#compare (CCEM, CEM)
# plt.show()