import numpy 
import pandas as pd
import glob, math, re
import matplotlib.pyplot as plt
import IsingMonteCarlo as im

dimensions=200
instances=1
iterations=5000

#run the model for different temperatures
temps=[0.1, 2, 3, 100, 300]
if(1):
	for i in temps:
		for nid in range(instances):
			print(i, nid)
			ising = im.IsingMonteCarlo(dimensions, i, iterations, nid)
			ising.run()
