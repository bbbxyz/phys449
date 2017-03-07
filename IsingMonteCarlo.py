'''
Montecarlo simulation of a 2D ising model

Saves one lattice state per row in the following format:
[S_00, S_01, ... S_NN, Temperature, Energy, Magnetization]
'''

import Constants as cst
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation



K_B = 1 #boltzmann const in m^2 kg s^-2 K^-1

class IsingMonteCarlo:
    
    def find_energy(self, p_num):
        m = [[0.5, 1, 0.5],[1, 0.5, 1],[0.5, 1, 0.5]]
        conv = signal.convolve2d(self.Grid[p_num], m, mode='same', boundary='wrap')
        energy = -self.J*0.5*np.sum(np.multiply(conv, self.Grid[p_num]))
        return energy
    
    def find_magnetization(self, p_num):
        mag = np.mean(self.Grid[p_num])
        return mag
     
    
    def state_change(self, Grid, p_num, energies, mags):
        rand_x = np.random.randint(0,self.N)
        rand_y = np.random.randint(0,self.N)
        self.Grid[self.p_num][:][:] = self.Grid[self.p_num-1][:][:]
        self.Grid[self.p_num][rand_y][rand_x] = -1 * self.Grid[self.p_num-1][rand_y][rand_x]
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
        self.mags[self.p_num] = self.find_magnetization(self.p_num)
        
    def run(self):
        for j in range(self.N):
            for i in range(self.N):
                var = np.random.randint(0,2)
                if var ==1:
                    self.Grid[self.p_num][j][i]=-1
                else:
                    self.Grid[self.p_num][j][i]=1       
        e0 = self.find_energy(self.p_num)
        m0 = self.find_magnetization(self.p_num)
        self.energies[self.p_num] = e0
        self.mags[self.p_num] = m0
        self.p_num+=1
        for i in range(1, self.num_permutations):
            self.state_change(self.Grid, i, self.energies, self.mags)
            self.p_num+=1
        
        data =[]
        for i, gr in enumerate(self.Grid[cst.skip:]):
            i+=cst.skip
            newrow = np.ravel(self.Grid[i]).tolist()
            newrow.append(self.T)
            newrow.append(self.energies[i])
            newrow.append(self.mags[i])
            data.append(newrow)
        data=np.array(data)
        
        #NOW SAVE self.Grid AND energies TO A TEXT FILE THAT CAN BE USED BY TENSOR FLOW
        #save as integers not floats to save space (we're only using integer values anyway)
        outfile = "data/%f.%i.csv" % (self.T, self.nid)
        np.savetxt(outfile,data , delimiter=",", fmt='%.2f')
        

    def __init__(self, N=20, T=1, num_permutations=1000, nid=0):
        cst.skip = 1000
        self.N = N
        self.nid = nid
        self.num_permutations = num_permutations + cst.skip
        self.p_num = 0      #current permutation number
        self.J = 1          #coupling coefficient
        self.h = 0.0        #B field - for now assume h = 0
        self.T = T          #temp in Kelvin
        self.Grid = np.zeros((self.num_permutations, self.N, self.N))
        self.energies = np.zeros(self.num_permutations)
        self.mags = np.zeros(self.num_permutations)
        

