import numpy as np

#constants for data generation
temps=np.arange(1e-8, 5, 0.01)
lattice_size = 16
iterations = 2000	#n. of iterations to run for
sampling_freq = 10 #save every n. iterations
skip = lattice_size**3		#n. of iterations to skip
instances = 5		#n. of each temperature to run
h = 0.0				#B field - for now assume h = 0
mu = 1.0       		#magnetic moment
K_B = 1.0 			#boltzmann const in m^2 kg s^-2 K^-1
J = 1.0       		#coupling coefficient
T_c = 2*J/(K_B * np.log(1+np.sqrt(2)))
