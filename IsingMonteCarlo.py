'''
Montecarlo simulation of a 2D ising model

Saves one lattice state per row in the following format:
[S_00, S_01, ... S_NN, Temperature, Energy, Magnetization]
'''

import Constants as cst
import numpy as np
from scipy import signal

class IsingMonteCarlo:
    
    def find_energy(self, grid):
        m = [[0.5, 1, 0.5],[1, 0, 1],[0.5, 1, 0.5]]
        conv = signal.convolve2d(grid, m, mode='same', boundary='wrap')
        energy = -0.5*cst.J*np.sum(np.multiply(conv, grid))\
                    - cst.h*cst.mu*np.sum(grid)
        return energy
    
    def find_magnetization(self, grid):
        mag = np.mean(grid)
        return mag
     
    
    def state_change(self, oldgrid):
        '''
        Creates a new state and returns lattice, energy, magnetization
        '''
        rand_x = np.random.randint(0,self.N)
        rand_y = np.random.randint(0,self.N)
        newgrid = np.array(oldgrid)
        newgrid[rand_y,rand_x] *= -1.0
        new_energy = self.find_energy(newgrid)
        old_energy = self.find_energy(oldgrid)
        En_change = new_energy - old_energy
        if En_change <= 0:
            #energy is decreasing - make a flip
            pass
        else:
            #energy is increasing - flip with probability exp(- [change in E] / K T)
            rand_num = np.random.rand()
            condition = np.exp(- En_change / (cst.K_B * self.T))
            if rand_num <= condition:
                pass
            else:
                new_energy = old_energy
                newgrid = oldgrid
        
        return newgrid, new_energy, self.find_magnetization(newgrid)
        
    def run(self):
        current_grid = np.ones((self.N,self.N))
        #generate a random seed lattice
        for j in range(self.N):
            for i in range(self.N):
                var = np.random.randint(0,2)
                if var ==1:
                    current_grid[j,i]=-1
        current_e = self.find_energy(current_grid)
        current_m = self.find_magnetization(current_grid)
        self.p_num+=1
        data =[]
        for i in range(1, self.num_permutations):
            current_grid, current_e, current_m = self.state_change(current_grid)
            self.p_num+=1
            if(self.p_num>cst.skip and self.p_num%cst.sampling_freq==0):
                newrow = np.ravel(current_grid).tolist()
                newrow.append(self.T)
                newrow.append(current_e)
                newrow.append(current_m)
                data.append(newrow)
        data=np.array(data)
        outfile = self.directory+"%.2f.%i.csv" % (self.T, self.nid)
        np.savetxt(outfile,data , delimiter=",", fmt='%.2f')
        

    def __init__(self, N=20, T=1.0, num_permutations=1000, nid=0, directory='data/'):
        self.directory = directory
        self.N = N
        self.nid = nid
        self.num_permutations = num_permutations + cst.skip
        self.p_num = 0      #current permutation number
        self.T = T          #temp in Kelvin
        self.Grid = np.zeros((int(self.num_permutations/cst.sampling_freq), self.N, self.N))
        self.energies = np.zeros(int(self.num_permutations/cst.sampling_freq))
        self.mags = np.zeros(int(self.num_permutations/cst.sampling_freq))
        

