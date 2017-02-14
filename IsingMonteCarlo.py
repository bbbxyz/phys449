import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tempfile import TemporaryFile

#define grid of size N by N
#values in grid can have values:
    # s = 1  --> spin up
    # s = -1 --> spin down
N = 20
x_len = N       #grid dimensions
y_len = N
num_permutations = 10000
fincrement=300 #take a snapshot every n frames
p_num = 0       #current permutation number
J = 1e-23          #coupling coefficient
h = 0           #B field - for now assume h = 0
T = 300  #temp in Kelvin
K_B = 1.38064852e-23        #boltzmann const in m^2 kg s^-2 K^-1

Grid = np.zeros((num_permutations, y_len, x_len))

for j in range(y_len):
    for i in range(x_len):
        var = np.random.randint(0,2)
        if var == 0:
            Grid[p_num][j][i]=-1
        if var == 1:
            Grid[p_num][j][i]=1        
#for N=3:
#[x=0,y=0][x=1,y=0][x=2,y=0]
#[x=0,y=1][x=1,y=1][x=2,y=1]
#[x=0,y=2][x=1,y=2][x=2,y=2]
#we now have a grid of random spins - this is the starting state of the lattice


                #find the energy of the lattice

#H = -J * SUM (s_i,j * s_i,j) - h * SUM(s_i,j)
#take h = 0

def find_energy(Grid, N, p_num):
    #compare horizontal neighbours for values of s_i*s_j
    #these make a matrix N-1 * N
    #for N=3:
    #[m=0,n=0][m=1,n=0]
    #[m=0,n=1][m=1,n=1]
    #[m=0,n=2][m=1,n=2]
    horizontal = np.zeros((N, N-1))
    for n in range(N):
        for m in range(N-1):
            horizontal[n][m] = Grid[p_num][n][m]*Grid[p_num][n][m+1]

    #compare vertical neighbours for values of s_i*s_j
    #these make a matrix N * N-1
    #for N=3:
    #[[m=0,n=0][m=1,n=0][m=2,n=0]
    #[m=0,n=1][m=1,n=1][m=2,n=1]
    vertical = np.zeros((N-1, N))
    for n in range(N-1):
        for m in range(N):
            vertical[n][m] = Grid[p_num][n][m]*Grid[p_num][n+1][m]

    #compare diagonal neighbours in NE-SW direction (i.e /)
    diagonal1 = np.zeros((N-1, N-1))
    for n in range(0, N-1):
        for m in range(0, N-1):
            diagonal1[n][m] = Grid[p_num][n+1][m]*Grid[p_num][n][m+1]

    
    #compare diagonal neighbours in NW-SE direction (i.e \)
    diagonal2 = np.zeros((N-1,N-1))
    for n in range(0, N-1):
        for m in range(0, N-1):
            diagonal2[n][m] = Grid[p_num][n][m]*Grid[p_num][n+1][m+1]

    #edge effects h and v
    edge1 = np.zeros((2*N))
    for n in range(0,N):
        edge1[n] = Grid[p_num, 0, n]*Grid[p_num, N-1, n] #top and bottom
        edge1[n+N] = Grid[p_num, n, 0]*Grid[p_num, n, N-1] #left and right
     
    #edge effects diagonal
    edge2 = np.zeros(4*N)
    for n in range(0,N-1):
        edge2[n] = Grid[p_num, 0, n] * Grid[p_num, N-1, n+1]  #top and bottom
        edge2[n+N] = Grid[p_num, 0, n+1] * Grid[p_num, N-1, n]
        edge2[n+2*N] = Grid[p_num, n, 0] * Grid[p_num, n+1, N-1] #left and right
        edge2[n+3*N] = Grid[p_num, n+1, 0] * Grid[p_num, n, N-1]

    edge2[N-1] = Grid[p_num, N-1, 0] * Grid[p_num, 0, N-1]        #opposite diagonals
    edge2[2*(N)-1] = Grid[p_num, 0, 0] * Grid[p_num, N-1, N-1]


    #add all the values in arrays to find the energy
    sum_h = np.sum(horizontal)
    sum_v = np.sum(vertical)
    sum_d1 = np.sum(diagonal1)
    sum_d2 = np.sum(diagonal2)
    sum_edge1 = np.sum(edge1)
    sum_edge2 = np.sum(edge2)
    energy = -J*(sum_h + sum_v + sum_edge1) + 0.5*J*(sum_d1 + sum_d2 + sum_edge2)
    return energy
 
#create array storing the energy for each state change
energies = np.zeros(num_permutations)
#find initial energy
e0 = find_energy(Grid, N, p_num)
energies[p_num] = e0

                #perform a random state change
                
def state_change(Grid, p_num, energies):
    rand_x = np.random.randint(0,x_len)
    rand_y = np.random.randint(0,y_len)
    #print([rand_x,rand_y])
    #print(Grid[p_num-1][:][:])
    Grid[p_num][:][:] = Grid[p_num-1][:][:]
    Grid[p_num][rand_y][rand_x] = -1 * Grid[p_num-1][rand_y][rand_x]
    #print(Grid[p_num][:][:])
    new_energy = find_energy(Grid, N, p_num)
    old_energy = find_energy(Grid, N, p_num-1)
    if new_energy <= old_energy:
        #energy is decreasing - make a flip
        energies[p_num] = new_energy
    else:
        #energy is increasing - flip with probability exp(- [change in E] / K T)
        rand_num = np.random.rand()
        En_change = new_energy - old_energy
        condition = np.exp(- En_change / (K_B * T))
        if rand_num <= condition:
            energies[p_num] = new_energy
        else:
            energies[p_num] = old_energy
            Grid[p_num] = Grid[p_num-1]
            

      
      
#def save(something):
    
                #perform num_permutations state changes
     
for i in range(1, num_permutations):
    state_change(Grid, i, energies)
    if i%(num_permutations/100) == 0:
        percentage_complete = i/num_permutations
        print('                     ', end='\r', flush=False)
        print(percentage_complete,"% complete!", end='\r', flush=False)

frame=0

fig = plt.figure()
im = plt.imshow(Grid[frame], animated=True, interpolation='none')

def updatefig(*args):
    global frame
    frame += fincrement
    im.set_array(Grid[frame])
    return im,
    
#ani = animation.FuncAnimation(fig, updatefig, interval=num_permutations/fincrement, blit=True)
#ani.save(str(T)+".mp4",fps=30)
#plt.show()

#print(energies)
#plt.imshow(Grid[0])
#plt.imshow(Grid[num_permutations-1])

x = np.arange(0, num_permutations, 1)

#plt.plot(x, 50*energies)
#plt.xlabel('Number of iterations')
#plt.ylabel('Energy (units??)')
#plt.show()

data =[]
for i, gr in enumerate(Grid):
    newrow = np.ravel(Grid[i]).tolist()
    newrow.append( energies[i])
    data.append(newrow)
data=np.array(data)

#NOW SAVE Grid AND energies TO A TEXT FILE THAT CAN BE USED BY TENSOR FLOW
        
#ask user to enter filename to save to, hopefully prevent us overwriting data!
outfile = str(T)+".csv"
np.savetxt(outfile,data , delimiter=",")

