'''
Script to generate data from the Ising MC model
'''

import Constants as cst
import numpy 
import pandas as pd
import glob, math, re, time
from multiprocessing import Process
import IsingMonteCarlo as im

t0 = time()
n_proc = 4
size = int(len(cst.temps)/float(n_proc))
if not os.path.exists("data"):
    os.makedirs("data")

def create_data(n_tot, n_id):
    temp_set = cst.temps[size*n_id:size*(n_id+1)]
    for i in temp_set:
        for nid in range(cst.instances):
            print(i, nid)
            ising = im.IsingMonteCarlo(cst.lattice_size, i, cst.iterations, nid)
            ising.run()

procs = []
for i in range(n_proc):
    procs.append(Process(target=create_data, args=(n_proc,i), daemon=True))

for proc in procs:
    proc.start()

for proc in procs:
    proc.join()
t1 = time()
print("Data generated. Time taken: %is" % (t1-t0))

