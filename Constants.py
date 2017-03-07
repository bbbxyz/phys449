'''
Constants are defined here
'''
import numpy as np

#constants for data generation
temps=np.arange(1e-8, 5, 1.0)
lattice_size = 32
iterations = 20000	#n. of iterations to run for
sampling_freq = 100 #save every n. iterations
skip = 50000 	#n. of iterations to skip
instances = 1		#n. of each temperature to run
h = 0.0				#B field - for now assume h = 0
mu = 1.0       		#magnetic moment
K_B = 1.0 			#boltzmann const in m^2 kg s^-2 K^-1
J = 1.0       		#coupling coefficient
T_c = 2*J/(K_B * np.log(1+np.sqrt(2)))
