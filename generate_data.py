'''
todo:
- move constants to Constants.py
'''

import Constants as cst
import numpy 
import pandas as pd
import glob, math, re
import matplotlib.pyplot as plt
import IsingMonteCarlo as im




#run the model for different temperatures

for i in cst.temps:
    for nid in range(cst.instances):
        print(i, nid)
        ising = im.IsingMonteCarlo(cst.lattice_size, i, cst.iterations, nid)
        ising.run()
