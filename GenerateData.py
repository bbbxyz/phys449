'''
Script to generate data from the Ising MC model

Constants are defined in Constants.py
'''

import Constants as cst
import time, os, sys
from multiprocessing import Process
import IsingMonteCarlo as im

n_proc = 4
size = int(len(cst.temps)/float(n_proc))
def create_data(n_tot, n_id, directory):
    temp_set = cst.temps[size*n_id:size*(n_id+1)]
    for i in temp_set:
        for nid in range(cst.instances):
            print(i, nid)
            ising = im.IsingMonteCarlo(cst.lattice_size, i, cst.iterations, nid, directory)
            ising.run()


if __name__ == '__main__':
    t0 = time.time()
    if not os.path.exists("data"):
        os.makedirs("data")

    if(len(sys.argv) < 2):
        print("No directory specified")
        exit(1)
    directory = sys.argv[1]  #get number of layers from command line
    procs = []
    for i in range(n_proc):
        procs.append(Process(target=create_data, args=(n_proc,i,directory,), daemon=True))

    for proc in procs:
        proc.start()

    for proc in procs:
        proc.join()
    t1 = time.time()
    print("Data generated. Time taken: %is" % (t1-t0))

