# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 18:39:45 2020

@author: RAJARSI
"""

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import timeit
#----------------------------------------------------------------------
##  semi-block codes################
#----------------------------------------------------------------------

@nb.njit(error_model="numpy")
def initialstate(N):   
    ''' generates a random spin configuration for initial condition'''
    state =2*np.random.randint(0,2, size=(N,N))-1
    return state

@nb.njit(error_model="numpy")
def calcEnergy(config,H):
    '''Energy of a given configuration'''
    N=len(config)
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i,j]
            nb = config[(i+1)%N, j] + config[i,(j+1)%N]  + config[(i-1)%N, j] + config[i,(j-1)%N] 
            energy += -nb*S - H*S
    return energy/(4.)

@nb.njit(error_model="numpy")
def calcMag(config,H):
    '''Magnetization of a given configuration'''
    mag = np.sum(config) + H
    return mag

@nb.njit(error_model="numpy")
def wolff(config,p,N):##wollff cluster implementation 
   ## the followng is the function for a sngle run of cluster making and flipping.
    que = []  ### "que" ; stores the list of coordinates of the neighbours aded to the cluster
    x0,y0 = np.random.randint(len(config)),np.random.randint(len(config)) ## a random spin is chosen at first
    que.append((x0,y0)) ## then added to the "que"


    while (len(que) > 0):## as mentioned in the documents I havesnt u , code must run untill there is nobody left to be added
        q = que[np.random.randint(len(que))] ## a random element from que is chosen

        neighbours = [((q[0]+1)%N,q[1]), ((q[0]-1)%N,q[1]), (q[0],(q[1]+1)%N), (q[0],(q[1]-1)%N) ] ## neighbours to the selected spin are considered
        for c in neighbours:
            if config[q]==config[c]  and c not in que and np.random.rand() < p:## process of adding spins to the que based on wolff's criteria if they have same spin
                que.append(c)


        config[q] *= -1 ## the spin'q' that was selected from the que is flipped so to avoid being selected in future
        que.remove(q)  ## the spin'q' is removed from the 'que'

    return config


@nb.njit(error_model="numpy",parallel=True)
def run_simulation(N,eqSteps,mcSteps,T,J,H):
    nt=T.shape[0]

    E,M,C,X = np.empty(nt), np.empty(nt), np.empty(nt), np.empty(nt)

    n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N) 

    #It looks like the calculation time heavily depends on the Temperature
    #shuffle the values to get the work more equaly splitted
    #np.random.shuffle isn't supported by Numba, but you can use a Python callback
    with nb.objmode(ind='int32[::1]'): 
        ind = np.arange(nt)
        np.random.shuffle(ind)

    ind_rev=np.argsort(ind)
    T=T[ind]

    for tt in nb.prange(nt):

        E1 = M1 = E2 = M2 = 0
        config = initialstate(N)
        iT=1.0/T[tt]; iT2=iT*iT;
        p = 1 - np.exp(-2*J*iT)

        for i in range(eqSteps):           # equilibrate
            config=wolff(config,p,N)       # Monte Carlo moves

        for i in range(mcSteps):
            config=wolff(config,p,N)            
            Ene = calcEnergy(config,H)     # calculate the energy
            Mag = abs(calcMag(config,H))   # calculate the magnetisation

            E1 = E1 + Ene
            M1 = M1 + Mag
            M2 = M2 + Mag*Mag 
            E2 = E2 + Ene*Ene


        E[tt] = n1*E1
        M[tt] = n1*M1
        C[tt] = (n1*E2 - n2*E1*E1)*iT2
        X[tt] = (n1*M2 - n2*M1*M1)*iT

        print ("Temp:",T[tt],":", E[tt], M[tt],C[tt],X[tt])

        #undo the shuffling
    return E[ind_rev],M[ind_rev],C[ind_rev],X[ind_rev]

#def control():
####################################
N  = 10       #N X N spin field
J  = 1
H  = 0
nt = 250
eqSteps = 80
mcSteps = 20

#############################################No change rquired here
T  = np.linspace(1.33, 4.8, nt) 

#You can measure the compilation time separately, it doesn't make sense to mix 
#runtime and compialtion time together.
#If the compilation time gets relevant it also make sense to use `cache=True`

start=timeit.default_timer()
E,M,C,X=run_simulation(N,eqSteps,mcSteps,T,J,H)
stop=timeit.default_timer()

print('Time taken for this simulation:: ', stop - start)


"""
1.17 s ± 74.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

#without parallelization
2.1 s ± 44 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

#without compilation
~130 s
"""

    
f = plt.figure(figsize=(25, 20)); # plot the calculated values    
plt.title("External Field :%5.2f"%(H))
sp =  f.add_subplot(3, 2, 1 );
plt.scatter(T, E, s=50, marker='o', color='Red')
plt.xlabel("Temperature (T)", fontsize=20);
plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');

sp =  f.add_subplot(3, 2, 2 );
plt.scatter(T, M, s=50, marker='o', color='Blue')
plt.xlabel("Temperature (T)", fontsize=20); 
plt.ylabel("Magnetization ", fontsize=20);   plt.axis('tight');

sp =  f.add_subplot(3, 2, 3 );
plt.scatter(T, C, s=50, marker='o', color='Red')
plt.xlabel("Temperature (T)", fontsize=20);  
plt.ylabel("Specific Heat ", fontsize=20);   plt.axis('tight');   

sp =  f.add_subplot(3, 2, 4 );
plt.scatter(T, X, s=50, marker='o', color='RoyalBlue')
plt.xlabel("Temperature (T)", fontsize=20); 
plt.ylabel("Susceptibility", fontsize=20);   plt.axis('tight');
 
sp = f.add_subplot(3 ,2 ,5);
plt.scatter(E, M,s=50, marker='+', color='Red') 
plt.xlabel("Energy ", fontsize=20);         plt.axis('tight');
plt.ylabel("Magnetization ", fontsize=20);   plt.axis('tight');

plt.show()



      
    









