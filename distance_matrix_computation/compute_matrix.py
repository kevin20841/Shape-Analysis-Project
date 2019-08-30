import sys
sys.path.append("../")
from joblib import Parallel, delayed, dump, load
import os, shutil
import shapedist
import numpy as np
import time
import subprocess
# host names
names = ["magenta", "benson", "jeeves", "renfield", "bunter"]
num_curves = 515
dirname = os.path.dirname(__file__)

filename = "cell_curves_ccem1e-4_srvf_noR.txt"

def printtime(t):
    if t < 60:
        print(str(t) + " seconds passed.", end='\r')
    if t >= 60:
        print(str(t//60) + " minutes, " + str(t % 60) + " seconds passed.", end='\r')

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
    curves = np.load("../data/marrow_cell_curves_full.npy", allow_pickle=True)

    # for i in range(100):
    #     curves[i] = curves_raw[i][0].T
    # for i in range(10):
    #     for j in range(2):
    #         curves[i * 2 + j] = curves_raw[i * 10 + j][0].T
    dump(curves, dfm)

def broadcast():
    N = len(names)
    for i in range(N):
        name = names[i]
        step = int(np.floor(num_curves/N))
        start = step * i
        end = min(num_curves, step *(i + 1))
        # print(start, end)
        if i == N -1:
            end = num_curves
        command = ["ssh","-f", name, "nohup","/users/kls6/anaconda3/envs/Shape-Analysis-Project/bin/python",
                   "/users/kls6/Shape-Analysis-Project/distance_matrix_computation/worker.py", str(start), str(end), "&>", "/dev/null"]
        print("Connected to", name)
        time.sleep(2)
        subprocess.Popen(command)


def wait():
    num = 0
    count = 0
    while num != len(names):
        # check every 20 seconds if all jobs have completed
        st = 5
        time.sleep(st)
        onlyfiles = os.listdir("./output") # dir is your directory path as string
        num = len(onlyfiles)
        count = count + 1
        printtime(st * count)
    print()
    f = open("time.txt", "a")
    f.write("Took " + str(count * st) + " seconds! "+ filename+"\n")
    f.close()
def assemble():
    output = np.zeros((num_curves, num_curves))
    # assemble all files
    data_matrices = os.listdir("./output")
    for name in data_matrices:
        raw_data = np.loadtxt(os.path.join(dirname, "./output/" + name))
        [start, end] = name.split("_")
        start = int(start)
        end = int(end)
        output[start:end] = raw_data
    if os.path.isdir("./output"):
        shutil.rmtree("./output")
    os.mkdir("./output")
    np.savetxt(filename, output)

def main():
    args = sys.argv
    if len(args) == 1:
        print("Caching")
        cache()
        broadcast()
        print("Waiting")
        wait()
        print("Finished!")
        assemble()
    else:

        if "-c" in args:
            cache()
        if "-w" in args:
            wait()
        if "-b" in args:
            broadcast()
        if "-a" in args:
            assemble()

if __name__ == "__main__":
    sys.exit(main())
