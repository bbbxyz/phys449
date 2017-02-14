import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tempfile import TemporaryFile

K_B = 1.38064852e-23  #boltzmann const in m^2 kg s^-2 K^-1

class IsingMonteCarlo:
	
	def find_energy(self, p_num):
		#compare horizontal neighbours for values of s_i*s_j
		#these make a matrix N-1 * N
		#for N=3:
		#[m=0,n=0][m=1,n=0]
		#[m=0,n=1][m=1,n=1]
		#[m=0,n=2][m=1,n=2]
		horizontal = np.zeros((self.N, self.N-1))
		for n in range(self.N):
			for m in range(self.N-1):
				horizontal[n][m] = self.Grid[p_num][n][m]*self.Grid[p_num][n][(m+1)%(self.N-1)]

		#compare vertical neighbours for values of s_i*s_j
		#these make a matrix N * N-1
		#for N=3:
		#[[m=0,n=0][m=1,n=0][m=2,n=0]
		#[m=0,n=1][m=1,n=1][m=2,n=1]
		vertical = np.zeros((self.N-1, self.N))
		for n in range(self.N-1):
			for m in range(self.N):
				vertical[n%self.N][m%self.N] = self.Grid[p_num][n%self.N][m%self.N]*self.Grid[p_num][((n+1)%(self.N-1))%self.N][m%self.N]

		#compare diagonal neighbours in NE-SW direction (i.e /)
		diagonal1 = np.zeros((self.N-1, self.N-1))
		for n in range(0, self.N-1):
			for m in range(0, self.N-1):
				diagonal1[n][m] = self.Grid[p_num][(n+1)%(self.N-1)][m]*self.Grid[p_num][n][(m+1)%(self.N-1)]

		
		#compare diagonal neighbours in NW-SE direction (i.e \)
		diagonal2 = np.zeros((self.N-1,self.N-1))
		for n in range(0, self.N-1):
			for m in range(0, self.N-1):
				diagonal2[n][m] = self.Grid[p_num][n][m]*self.Grid[p_num][(n+1)%(self.N-1)][(m+1)%(self.N-1)]

		#edge effects h and v
		edge1 = np.zeros((2*self.N))
		for n in range(0,self.N):
			edge1[n] = self.Grid[p_num, 0, n]*self.Grid[p_num, self.N-1, n] #top and bottom
			edge1[n+self.N] = self.Grid[p_num, n, 0]*self.Grid[p_num, n, self.N-1] #left and right
		 
		#edge effects diagonal
		edge2 = np.zeros(4*self.N)
		for n in range(0,self.N-1):
			edge2[n] = self.Grid[p_num, 0, n] * self.Grid[p_num, self.N-1, (n+1)%(self.N-1)]  #top and bottom
			edge2[n+self.N] = self.Grid[p_num, 0, (n+1)%(self.N-1)] * self.Grid[p_num, self.N-1, n]
			edge2[n+2*self.N] = self.Grid[p_num, n, 0] * self.Grid[p_num, (n+1)%(self.N-1), self.N-1] #left and right
			edge2[n+3*self.N] = self.Grid[p_num, (n+1)%(self.N-1), 0] * self.Grid[p_num, n, self.N-1]

		edge2[self.N-1] = self.Grid[p_num, self.N-1, 0] * self.Grid[p_num, 0, self.N-1]        #opposite diagonals
		edge2[2*(self.N)-1] = self.Grid[p_num, 0, 0] * self.Grid[p_num, self.N-1, self.N-1]


		#add all the values in arrays to find the energy
		sum_h = np.sum(horizontal)
		sum_v = np.sum(vertical)
		sum_d1 = np.sum(diagonal1)
		sum_d2 = np.sum(diagonal2)
		sum_edge1 = np.sum(edge1)
		sum_edge2 = np.sum(edge2)
		energy = -self.J*(sum_h + sum_v + sum_edge1) + 0.5*self.J*(sum_d1 + sum_d2 + sum_edge2)
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

		#NOW SAVE self.Grid AND energies TO A TEXT FILE THAT CAN BE USED BY TENSOR FLOW
				
		#ask user to enter filename to save to, hopefully prevent us overwriting data!
		outfile = "%i.csv" % (self.T)
		np.savetxt(outfile,data , delimiter=",")
		

	def __init__(self, N=20, T=1, num_permutations=1000):
		#define grid of size N by N
		#values in grid can have values:
			# s = 1  --> spin up
			# s = -1 --> spin down
		self.N = N
		self.num_permutations = num_permutations
		self.fincrement=10 #take a snapshot every n frames
		self.p_num = 0       #current permutation number
		self.J = 1e-23          #coupling coefficient
		self.h = 0           #B field - for now assume h = 0
		self.T = T  #temp in Kelvin
		self.Grid = np.zeros((num_permutations, self.N, self.N))
		#create array storing the energy for each state change
		self.energies = np.zeros(self.num_permutations)
		#find initial energy


