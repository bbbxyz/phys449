'''
Script to generate data from the Ising MC model
'''

import Constants as cst
import numpy 
import pandas as pd
import glob, math, re, time
import matplotlib.pyplot as plt
import IsingMonteCarlo as im


t0 = time.time()
#run the model for selected temperatures
for i in cst.temps:
    for nid in range(cst.instances):
        print(i, nid)
        ising = im.IsingMonteCarlo(cst.lattice_size, i, cst.iterations, nid)
ising.run()
t1 = time.time()
print("Data generated. Time taken: %is" % (t1-t0))
