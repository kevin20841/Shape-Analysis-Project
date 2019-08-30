import numpy as np
import editdistance
import matplotlib.pyplot as plt
from tqdm import trange
# TODO USE pyxDamerauLevenshtein!!

def compare(n, b):

    bf = b.flatten()
    nf = n.flatten()
    dperc = nf - bf
    # m = np.argmax(nf - bf)
    # i = m//515
    # j = m - m//515 * 515
    # print(m, m//515, m - m//515 * 515)
    # print(n[i][j], b[i][j] )

    for i in range(dperc.shape[0]):
        dperc[i] = 0 if bf[i] == 0 or nf[i] == 0 or bf[i] < 0.05 else (nf[i]-bf[i])/ bf[i]
    plt.figure();
    plt.scatter(np.arange(dperc.shape[0]), dperc, s=0.05, marker="o", c = "maroon")
    print(dperc[dperc > 0.5].shape[0] / dperc.shape[0])
    dist = []
    for i in range(1):
        bs = np.argsort(b[i])
        ns = np.argsort(n[i])
        dist.append(editdistance.distance(bs, ns))

    print(dist)
    # print(editdistance.distance(bs, ns))

DPEM = np.loadtxt("../cell_curves_uniformN_2_srvf_noR.txt")

CEM = np.loadtxt("../cell_curves_cem_srvf_noR.txt")

CCEM2e3 = np.loadtxt("../cell_curves_ccem_2e3_srvf_noR.txt")
CCEM1e4 = np.loadtxt("../cell_curves_ccem1e-4_srvf_noR.txt")



plt.figure(); plt.imshow(DPEM, cmap='hot', interpolation ="nearest")
plt.figure(); plt.imshow(CEM, cmap='hot', interpolation = "nearest")
plt.figure(); plt.imshow(CCEM2e3, cmap='hot', interpolation ="nearest")
plt.figure(); plt.imshow(CCEM1e4, cmap='hot', interpolation ="nearest")

compare(CCEM2e3, CEM)
#compare(CCEM, DPEM)
#compare (CCEM, CEM)
plt.show()