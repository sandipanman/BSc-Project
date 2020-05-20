# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:37:16 2020

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

size=5
steps=200


plot_en=[]
plot_mag=[]
mag=np.zeros([steps])
Energy=np.zeros([steps])
sp_heat=np.zeros([steps])
High_mag=1

temperature=1
spin_states=(np.ones([size,size]))    

for i in range(size):
        for j in range (size):
                spin_states[i][j]=(random.choice([-1,1]))

c=np.array([[np.arange(0,High_mag,0.1)],[np.arange(High_mag,0,-0.1)],[(-1)*np.arange(0,High_mag,0.1)],[np.arange(-High_mag,0,0.1)],[np.arange(0,High_mag,0.1)]])    
c=c.ravel()

for H in c:
    print(H)
    for step in range(steps+200):
         
         e=0
         for i in range(size):
            for j in range (size):
                e = U(spin_states, size, i, j,H)
                if e <= 0:
                    spin_states[i, j] *= -1
                elif np.exp((-1.0 * e)/temperature) > random.random():
                    spin_states[i, j] *= -1
            
          
         if(step>200):
             mag[step-200]=(np.sum(spin_states)) / (size ** 2)    
             
             for i in range(size):
                    for j in range (size):
                        
                        Energy[step-200]+=U(spin_states,size,i,j,H)
                
    plot_en.append(np.mean(Energy))
    plot_mag.append(np.mean(mag))
    
  
print(plot_en)
print(plot_mag)
plt.plot(c,plot_mag,'r-')


            
    
        
    
        
        




        