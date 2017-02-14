import numpy 
import pandas as pd
import glob, math, re
import matplotlib.pyplot as plt
import IsingMonteCarlo as im

dimensions=25
iterations=5000

#run the model for different temperatures
temps=[0.5, 1, 5, 100, 1000, 10000]
for i in temps:
	ising = im.IsingMonteCarlo(dimensions, i, iterations)
	ising.run()

#plots each dataset's energies and inital and final states
def plot_set(f, fig1, fig2):
	df = pd.read_csv(f)
	temp = str(f[:-4])+" K"
	energies = df.values[:,-1]
	spins = df.values[:,:-1]
	N = math.sqrt(len(spins[-1]))
	end = spins[-1].reshape((N,N))
	first = spins[0].reshape((N,N))
	
	plt.figure("spins")
	fig1.imshow(first,interpolation ='none')
	fig1.set_title(temp)
	fig1.axis('off')
	fig2.imshow(end,interpolation ='none')
	fig2.axis('off')
	
	plt.figure("energies")
	plt.plot(energies)




def tryint(s):
    try:
        return int(s)
    except:
        return s
def alphanum_key(s):
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

#sort the files in numerical order
datasets = []
files = glob.glob('*.csv')
files.sort(key=alphanum_key)
for file in files:
	f=open(file, 'r')
	datasets.append(file)
	f.close()
	

f, sp = plt.subplots(2, len(datasets), sharex=True, sharey=True)
plt.figure("energies")
for i, ds in enumerate(datasets):
	plot_set(ds, sp[0][i], sp[1][i])
	
plt.figure("energies")
plt.xlabel('Iterations')
plt.ylabel('Energy (J)')
plt.legend(labels=temps)

plt.show()
