 # -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 08:54:54 2020

@author: Sandipan
"""
import numpy as np
import random
import matplotlib.pyplot as plt


# reduced units:
#T(reduced) = kT/epsilon | r(reduced) = r/sigma | U(reduced) = U/epsilon
# General Parameters
DIM=3
npart=500
L=79.3
volume=L**DIM
density=npart/volume
print("volume = ", volume, " density = ", density,"Number of atoms =",npart)
print ("L is " ,L)
T = 8.5e-1; Nsteps = 100000; maxdr =0.01;printfeq=100;DIM=2
#System parameters
beta=1/T



def E(dr2):
    """Returns LJ 6-12 interaction energy for a particular distance between
       2 particles"""
        
    return  4*((dr2)**(-6) - (dr2)**(-3)) # r is given in unit of sigma. dr2 is distance^2

def P(x):
    """ gives boltzman factor for position at x"""
    return np.exp(-(beta*E(x))) 

def PBC(L,pos):       
    
    """PBC check for dim dimension system with equal length L in all dimension. 
    INPUT : position array,length,dimension
    OUTPUT: New position
    """
    for k in range(DIM):
            if (pos[k]>0.5):
                pos[k]=pos[k]-1
            if (pos[k]<-0.5):
                pos[k]=pos[k]+1
        
            
    return (pos)


def distance(current_position):
    """Takes the current position array of the configuration and finds out 
    distance between each pairs. Neglectd if distance > rcutoff
    INPUT: Array of current position of each particles
    OUTPUT: Array containing distances between each pair of LJ particles.
    """
    Distances=[]
    for i in range (npart):
        for j in range (i+1,npart):
            dr=(current_position[i]-current_position[j])*L
            dr2=np.dot(dr,dr)
            
            if (dr2!=0):
                Distances.append(dr2)
            
    return Distances



Energy=[0 for _ in range (Nsteps)]
Distances=[0 for _ in range (Nsteps)]
current_position=np.zeros([npart,DIM])


 
#------------------ Initialise the Setup with random position ------------

ip=-1
x=0
y=0
lim=int(np.sqrt(npart))+1
for i in range(0,lim):
    for j in range(0,lim):
        if(ip<npart):
            x=i*(1/lim)
            y=j*(1/lim)
            current_position[ip]=np.array([x,y])
            ip=ip+1
        else:
            break
MassCentre = np.sum(current_position,axis=0)/npart
current_position=current_position-MassCentre


Distances[0]=distance(current_position)
        


for i in Distances[0]:
   
    Energy[0]+=E(i)
print(Energy[0])


# -------------------------MC Simulation ----------------------------
rejected=0
for step in range(1,Nsteps):
    if (step%printfeq==0):
        print ("Completed ",step,"steps")
    trial_position=np.zeros([npart,DIM])
    
    trial_energy=0
    
    for i in range (npart):
        
        displacex=(random.uniform(0,1)-0.5)*maxdr 
        displacey=(random.uniform(0,1)-0.5)*maxdr 
      
        pos=current_position[i]+np.array([displacex,displacey])
        trial_position[i]=PBC(L,pos)
    Distances[step]=distance(trial_position)
    #print(trial_position)
   
    for i in ((Distances[step])):
        trial_energy+= E(i)
    if (trial_energy<Energy[step-1]):
        current_position=trial_position
        Energy[step]=trial_energy 
    else:
        delta=trial_energy-Energy[step-1]
        if (random.random()<P(delta)):
           
            current_position=trial_position
            Energy[step]=trial_energy 
           
        else:
            rejected+=1
            
            Energy[step]=Energy[step-1]
    if (step%10==0):
            print ( Energy[step])        
    
            
print(Energy)           
print ("Rejected moves ",rejected,"out of",Nsteps)
steps=np.arange(0,Nsteps)            
plt.figure(1)    
plt.plot(steps, Energy,'o',label='simulation result')
#plt.plot(steps, Distances,'b-',label='simulation result')
plt.xlabel(' steps')
plt.ylabel('Energy of configuration')
plt.show()


    
        