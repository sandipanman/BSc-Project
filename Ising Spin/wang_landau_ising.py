# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:07:40 2020

@author: Sandipan
"""


import matplotlib.pyplot as plt
import numpy as np
import numba as nb

Nitt=100000000   # Total number of Monte Carlo steps
N=16            # Linear size of 2D Ising model, lattice = N x N
flatness = 0.90  # The condition to reset the Histogarm when
                # min(Histogram) > average(Histogram)*flattness

N2=N*N          # Total number of lattice sites


@nb.njit(error_model="numpy")
def initialstate(N):   
    ''' generates a random spin configuration for initial condition'''
    config =2*np.random.randint(0,2, size=(N,N))-1
    return config

@nb.njit(error_model="numpy")           
def CEnergy(config):
    "Energy of a 2D Ising lattice"
    N=len(config)
    energy = 0
    for i in range(N):
        for j in range(N):
            S = config[i,j]
            nbr = config[(i+1)%N, j] + config[i,(j+1)%N]  + config[(i-1)%N, j] + config[i,(j-1)%N] 
            energy += -nbr*S 
    return int(energy/2.)

@nb.njit(error_model="numpy") 
def Thermod(T, lngE, Energies, E0,N2):
    "Thermodynamics using density of states"
    Z=0
    Ev=0
    E2v=0
    F = 0
    Entropy=0
    for i,E in enumerate(Energies):
        w = np.exp(lngE[i]-lngE[0] - (E+E0)/T) # density of states g(E)
        Z   += w                            # Partition Function
        Ev  += w * E                        # Avdrage energy using sum(g(i)E(i))
        E2v += w * E**2                     # Mean square energy using sum(g(i)E(i))                   
    F = -T*np.log(Z)                        # Free Energy
    Ev *= 1./Z                              # Normalisation
    Entropy = (Ev-F) /T                           
    cv = (E2v/Z - Ev**2)/T**2               # Specific Heat
    return (Ev/N2, cv/N2,F/N2,Entropy/N2 )


@nb.njit(error_model="numpy") 
def wang_landau(Nitt, N, N2, indE, E0, flatness,Energies):
    "Wang Landau algorithm in Python"
    # Ising lattice at infinite temperature
    config = initialstate(N)
    # Corresponding energy
    Ene = CEnergy(config)
    # Logarithm of the density of states log(g(E))
    lngE = np.ones(len(Energies))
    # Histogram
    Hist = np.zeros(len(Energies))  
    
    """ 
    modification factor which modifies g(E)
    according to the formula g(E) -> g(E)*f,
    or equivalently, lngE[i] -> lngE[i] + lnf
    """
    lnf = 1.0
    
    for itt in range (Nitt):
        
        i = np.random.randint(N)
        j = np.random.randint(N)
        S = config[i,j]
        nbr = config[(i+1)%N, j] + config[i,(j+1)%N]  + config[(i-1)%N, j] + config[i,(j-1)%N] 
        Enew = Ene + 2*S*nbr        # The energy of the tryed iteration
        P = np.exp(lngE[indE[Ene+E0]]-lngE[indE[Enew+E0]])  # Probability to accept according to Wang-Landau
        if P > np.random.random():  # Metropolis condition
            config[i,j] = -S        # step is accepted, update lattice
            Ene = Enew              #    and energy
            
        Hist[indE[Ene+E0]] += 1.    # Histogram is update at each Monte Carlo step!
        lngE[indE[Ene+E0]] += lnf   # Density of states is also modified at each step!
        if itt % 100 == 0:
            aH = np.sum(Hist)/(N2+0.0) # mean Histogram
            mH = np.min(Hist)          # minimum of the Histogram
            if mH > aH*flatness:    # Is the histogram flat enough?
                Hist = np.zeros(len(Hist)) # Resetting histogram
                lnf /= 2.                  # and reducing the modification factor
                print (itt, 'histogram is flatt', mH, aH, 'f=', np.exp(lnf))
    return (lngE, Hist)

if __name__ == '__main__':

    # Possible energies of the Ising model
    Energies = (4*np.arange(N2+1)-2*N2).tolist()
    #print(Energies)
    Energies.pop(1)   # Note that energies Emin+4 and Emax-4 
    Energies.pop(-2)  # are not possible, hence removing them!
    
    Energies = tuple(Energies)
    
    # Maximum energy
    E0 = Energies[-1]                         
    # Index array which will give us position in the Histogram array from knowing the Energy
    indE = -np.ones(E0*2+1, dtype=int)           
    for i,E in enumerate(Energies): 
        indE[E+E0]=i
    #print (indE)
    
    (lngE, Hist) = wang_landau(Nitt, N, N2, indE, E0, flatness,Energies)

    
    # Normalize the density of states, knowing that the lowest energy state is double degenerate
    # lgC = log( (exp(lngE[0])+exp(lngE[-1]))/4. )
    if lngE[-1]<lngE[0]:
        lgC = lngE[0] + np.log(1+ np.exp(lngE[-1]-lngE[0])) - np.log(4.)
    else:
        lgC = lngE[-1] + np.log(1+ np.exp(lngE[0]-lngE[-1])) - np.log(4.)
    lngE -= lgC
    for i in range(len(lngE)):
        if lngE[i]<0: lngE[i]=0
    # Normalize the histogram
    Hist *= len(Hist)/float(sum(Hist))
    
    
    plt.plot(Energies, lngE, '-o', label='log(g(E))')
    plt.plot(Energies, Hist, '-s', label='Histogram')
    plt.title ('Density of states')
    plt.ylabel('log (g(E))')
    plt.xlabel('Energy')
    plt.legend(loc='best')
    plt.show()
    
    
    Te = np.linspace(0.5,4.,300)
    Thm=[]
    for T in Te:
        Thm.append(Thermod(T, lngE, Energies, E0,N2))
    Thm = np.array(Thm)
    
    
    
    plt.figure()
    plt.title ( 'Energy variation with Temperature(in arbitary unit)')
    plt.plot(Te,Thm[:,0],'r+', label='E(T)')
    plt.ylabel('E/N')
    plt.xlabel('T')
    plt.legend(loc='best')
    plt.show()
    
    plt.figure()
    plt.title ( 'Specific Heat variation with Temperature(in arbitary unit)')
    plt.plot(Te,Thm[:,1],'b+', label='Cv(T)')
    plt.ylabel('Cv/N')
    plt.xlabel('T')
    plt.legend(loc='best')
    plt.show()

    
    plt.figure()
    plt.title ( 'Free Energy variation with Temperature(in arbitary unit)')
    plt.plot(Te,Thm[:,2],'r+', label='F(T)')
    plt.ylabel ('F/N')
    plt.xlabel('T')
    plt.legend(loc='best')
    plt.show()
    
    plt.figure()
    plt.title ( 'Entropy variation with Temperature(in arbitary unit)')
    plt.plot(Te,Thm[:,3],'b+', label='Entropy(T)')
    plt.ylabel ('Entropy/N')
    plt.xlabel('T')
    plt.legend(loc='best')
    plt.show()


