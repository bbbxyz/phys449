'''
todo:
- move constants to Constants.py
- shuffle datasets??
'''

import Constants as cst
import numpy 
import pandas as pd
import glob, math, re
import matplotlib.pyplot as plt
import IsingMonteCarlo as im

dimensions=cst.lattice_size
instances= 500
iterations=500

#run the model for different temperatures
temps=[ 0.5, 1, 2, 5,  ]

for i in temps:
    for nid in range(instances):
        print(i, nid)
        ising = im.IsingMonteCarlo(dimensions, i, iterations, nid)
        ising.run()
