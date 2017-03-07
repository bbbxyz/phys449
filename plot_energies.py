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
    mags = np.abs(np.mean(df.values[:,-1]))
    energies = df.values[:,-2]
    temps = df.values[-1,-3]
    temp = str(temps)+" K"
    spins = df.values[:,:-3]
    N = cst.lattice_size
    '''
    end = spins[-1].reshape((N,N))
    first = spins[0].reshape((N,N))
    plt.imshow(first,interpolation ='none')
    plt.imshow(end,interpolation ='none')
    '''
    plt.figure("energies")
    plt.plot(energies)
    plt.figure("magnetizations")
    plt.scatter(temps, mags)




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
#plt.legend(labels=datasets)

plt.show()
