# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:50:16 2020

@author: Sandipan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:37:16 2020

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
    return spin_states[x,y]*(find_neighbours(spin_states,size,x,y))-H*(np.sum(spin_states))+Q*(m**2)

#------------------Simulation Parameters--------------------
Q=30
size=5
steps=50
H=0
max_trials=100
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
                magnet=magnetisation(spin_states,size)
                e = U(spin_states, size, i, j,H,magnet)
                if e <= 0:
                    spin_states[i, j] *= -1
                elif np.exp((-1.0 * e)*beta) > random.random():
                    spin_states[i, j] *= -1
            
          
         if(step>100):
             mag[step-100]=magnetisation(spin_states,size)
             magnet=magnetisation(spin_states,size)
             for i in range(size):
                    for j in range (size):
                        
                        Energy[step-100]+=round(U(spin_states,size,i,j,H,magnet),2)
                
    plot_en.append(round(np.mean(Energy),2))
    values_mag.append(round(np.mean(mag),3))
    
unique,unique_indices,unique_counts=np.unique(values_mag,return_index=1,return_counts=1)  
x_values=np.arange(-1,1,0.001)
hist_bins=np.arange(-1,1,0.01)
plt.figure(figsize=(6,4),dpi=80, facecolor='w', edgecolor='b')
plt.xlabel('magnetisation')
plt.ylabel('Frequency')
plt.title('Histogram for magnetisation at temperature %f'%beta)        
sns.distplot(values_mag,bins=hist_bins,kde=0)


for i in np.arange(-1,1,0.001):
    if (round(i,3) in unique):
        prob_mag.append(unique_counts[np.where(unique==round(i,3))][0]/max_trials)
        
    else:
        
        prob_mag.append(0)
        
    
for i in (prob_mag):
    if (i!=0) :
        free_energy.append(-(1/beta)*np.log(i)-Q*i**2)
    else:
        free_energy.append(-Q*i**2)

print('Probability magnitude for each magnetisation :',prob_mag)  
print('Free Energy :',free_energy)  

plt.figure(figsize=(6,4),dpi=80, facecolor='w', edgecolor='b')
plt.xlabel('magnetisation')
plt.ylabel('$-kTlogP_i$')
plt.title('Free Energy surface')        
plt.plot(x_values,free_energy,'r-',label='Magnetic Field (H): %d \n Beta:%f'%(H,beta))
        



            
    
        
    
        
        




        