# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:50:16 2020

@author: Sandipan
"""


import numba as nb
import numpy as np
import matplotlib.pyplot as plt


@nb.njit(error_model="numpy")
def deltanrg(spin_states,i,j,k,H,Q):
    
    size= len(spin_states)

    #Find the nearest neighbours
    nbr=(spin_states[i,j-1,k]+spin_states[i,(j+1)%size,k]+spin_states[i-1,j,k]+
        spin_states[(i+1)%size,j,k]+spin_states[i,j,k-1]+spin_states[i,j,(k+1)%size])
   
    m_old = np.mean(spin_states)
    energy=2*spin_states[i,j,k]*nbr+2*H*spin_states[i,j,k] #Change in energy due to spin flip
    
    spin_states[i, j, k] *= -1
    m_new = np.mean(spin_states)
    
    energy=energy+Q*((m_new)**2-(m_old)**2)  #Correction to include change inbias effect
    
    return energy




@nb.njit()
def mc_ising(size,num_steps,H,kT,Q):
    values_mag=[]
    
    
    spin_states=np.ones((size,size,size))
    nspins = size*size*size
    
    for step in range(num_steps): 
        for _ in range (nspins): 
            i=np.random.randint(size)
            j=np.random.randint(size)
            k=np.random.randint(size)
    
             
            dE=deltanrg(spin_states,i,j,k,H,Q) # Find initial configuration energy
            if np.exp((-1.0 * dE)/kT) < np.random.random():
                spin_states[i, j, k] *= -1             #FlipRejected
                
      
        magnet = np.mean(spin_states)
        values_mag.append(magnet)
    
    return values_mag


#------------------Simulation Parameters--------------------
Q = 0
size=10
num_steps=100000
H=0.3
num_bins=50


#----------------------------------------------------------    
plt.figure(num=1,figsize=(6,4),dpi=80, facecolor='w', edgecolor='b')
plt.xlabel('magnetisation')
plt.ylabel('$-kTlogP_i$')

    
for kT in  ([4.6]):


    values_mag =mc_ising(size,num_steps,H,kT,Q)   
    histo,bin_edges=np.histogram(values_mag,bins=num_bins,density=1)
    """
    plt.figure(num=0,figsize=(6,4),dpi=80, facecolor='w', edgecolor='b')
    plt.xlabel('magnetisation')
    plt.ylabel('Frequency')
    plt.title('Histogram for magnetisation at kT %.1f H %.1f'%(kT,H))        
    plt.hist(values_mag,bins=num_bins)
    """
        
    
    
    
    # Array of meean magnetusation in each bin
    mean_mag =[(bin_edges[i+1]+bin_edges[i])/2 for i in range(len(bin_edges)-1)] 
    #Bias potential
    
    W=np.array([Q*i**2 for i in mean_mag])
    print(W)
    
    F_corrected=[]
    F=[]
    
    for i in range(len(W)):
        if histo[i]!=0: 
    
            f= -kT * np.log(histo[i]) 
            F.append(f)                    #Free energy without bias correction
            F_corrected.append( f - W[i] ) #Correction for bias potential
          
   
    F_corrected=np.array(F_corrected)
    F_corrected=F_corrected -(np.nanmin(F_corrected, axis=0))
    
            
           
    #plt.plot(bin_edges[:-1][histo != 0],-kT * np.log(histo[histo != 0]), "o-",label = 'kT %.1f'%(kT))
    
    #plt.legend(loc="best")
    #plt.show()
    
    
    
    

    plt.figure(num=2,figsize=(6,4),dpi=80, facecolor='w', edgecolor='b')
    plt.xlabel('magnetisation')
    plt.ylabel('$-kTlogP_i$')
    plt.title("Free Energy surface sampled external field %d "%Q)        
    plt.plot(bin_edges[:-1][histo != 0],F_corrected, "bo-",label = 'kT %.1f H %.1f'%(kT,H))
    plt.legend(loc="best")
    plt.show()



            
    
        
    
        
        




        