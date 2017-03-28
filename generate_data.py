'''
Script to generate data from the Ising MC model
'''

import Constants as cst
import numpy 
import pandas as pd
import glob, math, re
from multiprocessing import Process
import matplotlib.pyplot as plt
import IsingMonteCarlo as im

#need to clean this up
size = int(len(cst.temps)/float(2))
def create_dataa():
	for i in reversed(cst.temps[:size]):
		for nid in range(cst.instances):
			print(i, nid)
			ising = im.IsingMonteCarlo(cst.lattice_size, i, cst.iterations, nid)
			ising.run()
def create_datab():
	for i in reversed(cst.temps[size:]):
		for nid in range(cst.instances):
			print(i, nid)
			ising = im.IsingMonteCarlo(cst.lattice_size, i, cst.iterations, nid)
			ising.run()

creator_thread1 = Process(target=create_dataa, daemon=True)
creator_thread2 = Process(target=create_datab, daemon=True)

creator_thread1.start()

creator_thread2.start()

creator_thread1.join()
creator_thread2.join()
