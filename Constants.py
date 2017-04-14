'''
Constants for data to be generated and analyzed
'''
import numpy as np

#range of temperatures to use
min_temp = 0
max_temp = 5
temp_step = 0.05

lattice_size =32
iterations = 8000	  #n. of iterations to run for
sampling_freq = 10  #save every n. iterations
skip = 10000        #n. of iterations to skip
instances = 2		  #n. of runs at each temperature

h = 0.0				  #external magnetic field 
mu = 1.0       	  #magnetic moment
K_B = 1.0 			  #boltzmann const in m^2 kg s^-2 K^-1
J = 1.0       		  #coupling coefficient

temps = np.arange(1e-8+min_temp, max_temp, temp_step )
T_c = 2*J/(K_B * np.log(1+np.sqrt(2)))
