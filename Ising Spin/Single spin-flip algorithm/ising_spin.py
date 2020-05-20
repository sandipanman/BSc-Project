# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 08:10:50 2020

@author: Sandipan
"""

import numpy as np
import random
import matplotlib.pyplot as plt

def find_neighbours(spin_states,size,x,y):
    left=spin_states[x,y-1]
    right=spin_states[x,(y+1)%size]
    top=spin_states[x-1,y]
    bottom=spin_states[(x+1)%size,y]
    tot_spin=left+right+top+bottom 
    
    return (tot_spin)


def U(spin_states,size,x,y,H):
    return spin_states[x,y]*(find_neighbours(spin_states,size,x,y))+H*(np.sum(spin_states))

    
size=10
steps=100
#H=0.1    

spin_states=np.random.choice([1, -1], size=(size, size))

plt.figure(1)
plt.xlabel('Temperature')
plt.ylabel('Magnetisation')
plt.title('Variation of Critical Temperature with external magnetic field')

for H in ([0]):
    plot_en=[]
    plot_mag=[]
    mag=np.zeros([steps])
    Energy=np.zeros([steps])
    sp_heat=np.zeros([steps])
    sigma=[]
    
    spin_states=np.random.choice([1, -1], size=(size, size))
    for temperature in np.arange(0.1,4,0.1):
        
        
        for step in range(steps+50):
             
             e=0
             for i in range(size):
                for j in range (size):
                    e = U(spin_states, size, i, j,H)
                    if e <= 0:
                        spin_states[i, j] *= -1
                    elif np.exp((-1.0 * e)/temperature) > random.random():
                        spin_states[i, j] *= -1
                
              
             if(step>50):
                 mag[step-50]=abs(np.sum(spin_states)) / (size ** 2)    
                 
                 for i in range(size):
                        for j in range (size):
                            
                            Energy[step-50]+=U(spin_states,size,i,j,H)
        sigma.append((np.var(Energy))/temperature**2)            
        plot_en.append(np.mean(Energy)/(size)**2)
        plot_mag.append(np.mean(mag))
        
    print(plot_en)
    print(plot_mag)
    dE=np.diff(plot_en)
    dEdT=dE/0.1
    plt.plot(np.arange(0.1,4,0.1),plot_mag,'-',label='External Magnetic Field: %f'%H)
print(sigma)    
plt.legend()    
plt.show()

plt.figure()
plt.plot(np.arange(0.1,4,0.1), plot_en,'-',label='External Magnetic Field: %f'%H)    
plt.plot(np.arange(0.1,4,0.1), sigma,'+',label='External Magnetic Field: %f'%H)    

            
    
        
    
        
        




        