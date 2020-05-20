# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:22:24 2020

@author: Sandipan
"""

## lj.py: a Monte Carlo simulation of the Lennard-Jones fluid, in the NVT ensemble.
import random
import sys
import numpy as np
import copy
import os
from numba import jit


# Simulation Parameters
N       = 2
sigma   = 1
epsilon = 1
trunc   = 3*sigma
truncsq = trunc**2
steps   = 10000000
temp    = 8.5e-1
density = 1.0e-3
DIM     =3
maxdr   =1
L =  (N/density)**(1.0/3.0)
halfL = L/2
particles = np.zeros([N,DIM])
phicutoff = 4.0/(trunc**12)-4.0/(trunc**6)


# Some helper functions
@jit(nopython=True)
def wrap(pos):
    '''Apply perodic boundary conditions.'''
    for k in range(DIM):
            if (pos[k]>0.5*L):
                pos[k]=pos[k]-L
            if (pos[k]<-0.5*L):
                pos[k]=pos[k]+L
        
            
    return (pos)    
    
@jit(nopython=True)
def distancesq(particle1, particle2):
    '''Gets the squared distance between two particles, applying the minimum image convention.'''
    # Calculate distances
    dx = particle1[0]-particle2[0]
    dy = particle1[1]-particle2[1]
    dz = particle1[2]-particle2[2]

    # Minimum image convention
    if dx > halfL:
        dx -= L
    elif dx < -halfL:
        dx += L
    if dy > halfL:
        dy -= L
    elif dy < -halfL:
        dy += L
    if dz > halfL:
        dz -= L
    elif dz < -halfL:
        dz += L

    return dx**2+dy**2+dz**2

@jit(nopython=True)
def energy(particles):
    '''Gets the energy of the system'''
    energy = 0
    for particle1 in range(0, len(particles)-1):
        for particle2 in range(particle1+1, len(particles)):
            dist = distancesq(particles[particle1], particles[particle2])
            if dist <= truncsq:
                energy += 4*(1/dist**6)-(1/dist**3)
    return energy

@jit(nopython=True)
def particleEnergy(particle, particles, p):
    '''Gets the energy of a single particle.'''
    part_energy = 0
   
    for i in range (len(particles)):
        if i != p:
            dist = distancesq(particle, particles[i])
            if dist <= truncsq:
                part_energy += 4*(1/dist**6)-(1/dist**3) - phicutoff
     
    return part_energy

def writeEnergy(step, en, isRej):
    '''Writes the energy to a file.'''
    
    with open('energy2.txt', 'a') as f:
        f.write('{0} {1} {2}\n'.format(step, en,isRej))

# Clear files if they already exist.
if os.path.exists('energy2'):
    os.remove('energy2')

# Initialize the simulation box:
for i in range(0, N):
    x_coord = random.uniform(0, L)
    y_coord = random.uniform(0, L)
    z_coord = random.uniform(0, L)
    particles[i]=np.array([x_coord, y_coord, z_coord])

# Calculate initial energy
en = energy(particles)
rejected=0
# MC
for step in range(0, steps):
    if (step%1000==0):
        sys.stdout.write("\rStep: {0} Rejected: {1}   Energy:  {2}".format(step, rejected, en))
        sys.stdout.flush()
    
    # Choose a particle to move at random.
    p = random.randint(0, N-1)
    isRej='Accepted'
    # Move particle and evaluate energy
    this_particle = copy.deepcopy(particles[p])
    #prev_E = particleEnergy(this_particle, particles, p)
    prev_E = energy(particles)
    this_particle[0] += random.uniform(-1, 1)*maxdr
    this_particle[1] += random.uniform(-1, 1)*maxdr
    this_particle[2] += random.uniform(-1, 1)*maxdr
    this_particle = wrap(this_particle)
    #new_E = particleEnergy(this_particle, particles, p)
    new_E = energy(particles)
    deltaE = new_E - prev_E

    # Acceptance rule enforcing Boltzmann statistics
    if deltaE < 0:
        particles[p] = this_particle
        en += deltaE
    elif(np.exp(-deltaE/temp) > random.random()):
            particles[p] = this_particle
            en += deltaE
    else:
        rejected+=1
        isRej='Rejected'
    try:        
        writeEnergy(str(step), str(en), str(isRej))
    except:
        pass