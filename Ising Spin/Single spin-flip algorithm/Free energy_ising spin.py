# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:50:16 2020

@author: Sandipan
"""



import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

def find_neighbours(spin_states,size,x,y):
    left=spin_states[x,y-1]
    right=spin_states[x,(y+1)%size]
    top=spin_states[x-1,y]
    bottom=spin_states[(x+1)%size,y]
    tot_spin=left+right+top+bottom 
    
    return (tot_spin)

def magnetisation(spin_states,size):
    return np.sum(spin_states)/size**2

def U(spin_states,size,x,y,H,m):
    return spin_states[x,y]*(find_neighbours(spin_states,size,x,y))-H*(np.sum(spin_states))+Q*((m)**2)

#------------------Simulation Parameters--------------------
Q=0
size=5
steps=10
H=0
max_trials=50
beta=1

#----------------------------------------------------------    

plot_en=[]
prob_mag=[]
values_mag=[]
mag=np.zeros([steps])
Energy=np.zeros([steps])
free_energy=[]
spin_states=(np.ones([size,size]))



for trials in range(max_trials):
    
    spin_states=np.random.choice([1, -1], size=(size, size))
    
    for step in range(steps+100): 
         
         e=0
         for i in range(size):
            for j in range (size):
                magnet=magnetisation(spin_states,size)   # Find magnetisation
                e = U(spin_states, size, i, j,H,magnet)  # Find energy
                if e <= 0:
                    spin_states[i, j] *= -1
                elif np.exp((-1.0 * e)*beta) > random.random():
                    spin_states[i, j] *= -1
            
          
         if(step>100): # Production Steps
             magnet=magnetisation(spin_states,size)
             mag[step-100]=magnet
             
             for i in range(size):
                    for j in range (size):
                        
                        Energy[step-100]+=round(U(spin_states,size,i,j,H,magnet),2)
                
    plot_en.append(round(np.mean(Energy),2))
    values_mag.append((magnet))
    
hist,bin_edges=np.histogram(values_mag,bins=100,density='True')
plt.figure(figsize=(6,4),dpi=80, facecolor='w', edgecolor='b')
plt.xlabel('magnetisation')
plt.ylabel('Frequency')
plt.title('Histogram for magnetisation at beta %f'%beta)        
sns.distplot(hist,bins=bin_edges,kde=0)



F=np.array([-(np.log(i))*(1/beta) for i in hist])  #Free energy
#Bias potential
W=np.array([Q*((bin_edges[i+1]-bin_edges[i])/2) for i in range(len(bin_edges)-1)])
F=F-W
print (F)
print (len(bin_edges))

print (bin_edges)


plt.figure(figsize=(6,4),dpi=80, facecolor='w', edgecolor='b')
plt.xlabel('magnetisation')
plt.ylabel('$-kTlogP_i$')
plt.title('Free Energy surface')        
sns.distplot(F,bins=bin_edges,label='Magnetic Field (H): %d \n Beta:%f'%(H,beta))
        



            
    
        
    
        
        




        