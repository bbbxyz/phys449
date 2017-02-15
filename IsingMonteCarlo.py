import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tempfile import TemporaryFile

K_B = 1 #boltzmann const in m^2 kg s^-2 K^-1

class IsingMonteCarlo:
	
	def find_energy(self, p_num):
		m = [[0, 1, 0],[1, 0, 1],[0, 1, 0]]
		conv = signal.convolve2d(self.Grid[p_num], m, mode='same', boundary='wrap')
		energy = -self.J*np.sum(np.multiply(conv, self.Grid[p_num]))
		return energy
	 
	
	def state_change(self, Grid, p_num, energies):
		rand_x = np.random.randint(0,self.N)
		rand_y = np.random.randint(0,self.N)
		#print(self.Grid[self.p_num-1][rand_y][rand_x])
		self.Grid[self.p_num][:][:] = self.Grid[self.p_num-1][:][:]
		self.Grid[self.p_num][rand_y][rand_x] = -1 * self.Grid[self.p_num-1][rand_y][rand_x]
		#print(self.Grid[self.p_num][rand_y][rand_x])
		new_energy = self.find_energy(self.p_num)
		old_energy = self.find_energy(self.p_num-1)
		if new_energy <= old_energy:
			#energy is decreasing - make a flip
			self.energies[self.p_num] = new_energy
		else:
			#energy is increasing - flip with probability exp(- [change in E] / K T)
			rand_num = np.random.rand()
			En_change = new_energy - old_energy
			condition = np.exp(- En_change / (K_B * self.T))
			if rand_num <= condition:
				self.energies[self.p_num] = new_energy
			else:
				self.energies[self.p_num] = old_energy
				self.Grid[self.p_num] = self.Grid[self.p_num-1]
				

	def run(self):
		for j in range(self.N):
			for i in range(self.N):
				var = np.random.randint(0,2)
				if var ==1:
					self.Grid[self.p_num][j][i]=-1
				else:
					self.Grid[self.p_num][j][i]=1       
		e0 = self.find_energy(self.p_num)
		self.energies[self.p_num] = e0
		self.p_num+=1
		for i in range(1, self.num_permutations):
			self.state_change(self.Grid, i, self.energies)
			self.p_num+=1
			if i%(self.num_permutations/100) == 0:
				percentage_complete = i/self.num_permutations
				print('                     ', end='\r', flush=False)
				print(percentage_complete,"% complete!", end='\r', flush=False)
		frame=0
		#fig = plt.figure()
		#im = plt.imshow(self.Grid[frame], animated=True, interpolation='none')

		def updatefig(*args):
			global frame
			frame += fincrement
			im.set_array(self.Grid[frame])
			return im,
			
		#ani = animation.FuncAnimation(fig, updatefig, interval=self.num_permutations/fincrement, blit=True)
		#ani.save(str(T)+".mp4",fps=30)

		'''
		#print(energies)
		plt.imshow(self.Grid[0])
		plt.imshow(self.Grid[self.num_permutations-1])

		x = np.arange(0, self.num_permutations, 1)

		plt.plot(x, 50*self.energies)
		plt.xlabel('Number of iterations')
		plt.ylabel('Energy (units??)')
		#plt.show()
		'''
		data =[]
		for i, gr in enumerate(self.Grid):
			newrow = np.ravel(self.Grid[i]).tolist()
			newrow.append( self.energies[i])
			data.append(newrow)
		data=np.array(data)
		#print(data[0])

		#NOW SAVE self.Grid AND energies TO A TEXT FILE THAT CAN BE USED BY TENSOR FLOW
		#save as integers not floats to save space (we're only using integer values anyway)
		outfile = "data/%i.%i.csv" % (self.T, self.nid)
		np.savetxt(outfile,data , delimiter=",", fmt='%i')
		

	def __init__(self, N=20, T=1, num_permutations=1000, nid=0):
		#define grid of size N by N
		#values in grid can have values:
			# s = 1  --> spin up
			# s = -1 --> spin down
		self.N = N
		self.nid = nid
		self.num_permutations = num_permutations
		self.fincrement=10 #take a snapshot every n frames
		self.p_num = 0       #current permutation number
		self.J = 1 #coupling coefficient
		self.h = 0.0           #B field - for now assume h = 0
		self.T = T  #temp in Kelvin
		self.Grid = np.zeros((num_permutations, self.N, self.N))
		#create array storing the energy for each state change
		self.energies = np.zeros(self.num_permutations)
		#find initial energy


