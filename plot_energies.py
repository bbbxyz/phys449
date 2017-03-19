import numpy 
import pandas as pd
import glob, math, re
import matplotlib.pyplot as plt
import IsingMonteCarlo as im
import Constants as cst
import numpy as np

#plots each dataset's energies and inital and final states
def plot_set(f):
    df = pd.read_csv(f)
    #mags = np.abs(df.values[:,-1])
    #temps = df.values[:,-3]
    #temp = str(temps[0])+" K"
    energies = df.values[:,-2]
    energies_m = np.mean(energies)
    energies_e = np.std(energies)
    mags = np.mean(np.abs(df.values[:,-1]))
    mags_e = np.std(df.values[:,-1])
    temps = df.values[-1,-3]
    temp = str(temps)+" K"
    spins = df.values[:,:-3]
    iterations = np.arange(cst.skip+cst.sampling_freq,\
			cst.skip+cst.iterations, cst.sampling_freq)
    N = cst.lattice_size
    '''
    plt.figure("spins")
    end = spins[-1].reshape((N,N))
    first = spins[0].reshape((N,N))
    plt.subplot(211), plt.imshow(first,interpolation ='none')
    plt.subplot(212), plt.imshow(end,interpolation ='none')
    plt.show()
    '''
    
    plt.figure("energies")
    plt.plot(iterations, energies)
    plt.figure("magnetizations")
    plt.subplot(211), plt.errorbar(temps, mags, yerr=mags_e, color='k')
    plt.plot((cst.T_c, cst.T_c), (0, 1), 'r--')
    plt.subplot(212), plt.errorbar(temps, energies_m, yerr=energies_e, color='k')
    plt.plot((cst.T_c, cst.T_c), (-4000,0), 'r--')



def tryint(s):
    try:
        return int(s)
    except:
        return s
def alphanum_key(s):
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

print("Sorting datasets...")
#sort the files in numerical order
datasets = []
files = glob.glob('data/*.csv')
files.sort(key=alphanum_key)
for file in files:
    f=open(file, 'r')
    datasets.append(file)
    f.close()

print("Plotting datasets...")
for i, ds in enumerate(datasets):
    plot_set(ds)
    
plt.figure("energies")
plt.xlabel('Iterations')
plt.ylabel('Energy')
plt.figure("magnetizations")
plt.xlabel('Temperature (K)')
plt.subplot(211), plt.ylabel('Average Magnetization per Site')
plt.subplot(212), plt.ylabel('Average Lattice Energy')
#plt.legend(labels=datasets)

plt.show()
