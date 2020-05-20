# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 12:10:42 2020

@author: Sandipan
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit,prange,objmode
from mpl_toolkits.mplot3d import Axes3D


# All globals are compile time constants
# recompilation needed if you change this values


# Setting up the simulation
NSteps   = 3500  # Number of steps
deltat   = 0.005  # Time step in reduced time units
temp     = 0.851  #Reduced temperature
DumpFreq = 100    # Save the position to file every DumpFreq steps
epsilon  = 1.0    # LJ parameter for the energy between particles
DIM      = 3
N        = 500
density  = 0.776
Rcutoff  = 3



#----------------------Function Definitions---------------------

#------------------Initialise Configuration--------


@jit(nopython=True,error_model="numpy",parallel=True)
def initialise_config(N,DIM,density,temp):
    sigma=np.sqrt(temp) #sqrt(kT/m)
    velocity = (np.random.normal(0.0,sigma,(N,DIM))-0.5)


    # Set initial momentum to zero
    COM_V = np.sum(velocity)/N     #Center of mass velocity
    velocity = velocity - COM_V    # Fix any center-of-mass drift

    # Calculate initial kinetic energy
    k_energy=0
    for i in prange (N):
        k_energy+=np.dot(velocity[i],velocity[i])
    vscale=np.sqrt(DIM*temp*N/k_energy)
    velocity*=vscale

    
    #Present coordintes of particles Initialize with zeroes
    coords = np.zeros((N,DIM))

    # Get the cooresponding box size
    L = (N/density)**(1.0/DIM)

    """ Find the lowest perfect cube greater than or equal to the number of
     particles"""
    nCube = 2

    while (nCube**3 < N):
        nCube = nCube + 1



    # Assign particle positions
    ip=-1
    x=0
    y=0
    z=0
    
    d=(L/nCube)
    for i in range(0,nCube):
        for j in range(0,nCube):
            for k in range(0,nCube):
                if(ip<N):
                    x=(i+0.5)*d
                    y=(j+0.5)*d
                    z=(k+0.5)*d
                    coords[ip]=np.array([x,y,z])
                    ip=ip+1
                else:
                    break
    return (coords/L),velocity,L        #Return coordinates in box scaled units


@jit(nopython=True)
def wrap(pos,L):
    '''Apply perodic boundary conditions.'''

    #correct array indexing
    for i in range (len(pos)):
        for k in range(DIM):
                if (pos[i,k]>0.5):
                    pos[i,k]=pos[i,k]-1
                if (pos[i,k]<-0.5):
                    pos[i,k]=pos[i,k]+1


    return (pos)    


@jit(nopython=True,error_model="numpy")
def LJ_Forces(pos,acc,epsilon,L,DIM,N):
    # Compute forces on positions using the Lennard-Jones potential
    # Uses double nested loop which is slow O(N^2) time unsuitable for large systems
    Sij = np.zeros(DIM) # Box scaled units
    Rij = np.zeros(DIM) # Real space units

    #Set all variables to zero
    ene_pot = 0
    acc = acc*0
    virial=0.0

    # Loop over all pairs of particles
    for i in range(N-1):
        for j in range(i+1,N):  #i+1 to N ensures we do not double count
            Sij = pos[i]-pos[j] # Distance in box scaled units
            for l in range(DIM): # Periodic interactions
                if (np.abs(Sij[l])>0.5):
                    Sij[l] = Sij[l] - np.copysign(1.0,Sij[l]) # If distance is greater than 0.5  (scaled units) then subtract 0.5 to find periodic interaction distance.

            Rij   = L*Sij           # Scale the box to the real units in this case reduced LJ units
            Rsqij = np.dot(Rij,Rij) # Calculate the square of the distance

            if(Rsqij < Rcutoff**2):
                # Calculate LJ potential inside cutoff
                # We calculate parts of the LJ potential at a time to improve the efficieny of the computation (most important for compiled code)
                rm2      = 1.0/Rsqij # 1/r^2
                rm6      = rm2**3    # 1/r^6
                rm12     = rm6**2    # 1/r^12
                forcefact= rm2*(rm12-0.5*rm6) 
                phi      = 4*(rm6**2-rm6)

                ene_pot+=phi # Accumulate energy

                

                virial     = virial-forcefact*Rsqij # Virial is needed to calculate the pressure
                acc[i]     = acc[i]+forcefact*Sij # Accumulate forces
                acc[j]     = acc[j]-forcefact*Sij # (Fji=-Fij)
    #If you want to get get the best performance, sum directly in the loop intead of 
    #summing at the end np.sum(ene_pot)
    return 48*acc, (ene_pot)/N, -virial/DIM # return the acceleration vector, potential energy and virial coefficient


@jit(nopython=True,error_model="numpy",parallel=True)
def Calculate_Temperature(vel,L,DIM,N):

    ene_kin = 0.0

    for i in prange(N):
        real_vel = L*vel[i]
        ene_kin = ene_kin + 0.5*np.dot(real_vel,real_vel)

    ene_kin_aver = 1.0*ene_kin/N
    temperature = 2.0*ene_kin_aver/DIM

    return ene_kin_aver,temperature


# Main MD loop
@jit(nopython=True,error_model="numpy")
def main(NSteps,production,deltat,temp,DumpFreq,epsilon,DIM,N,Rcutoff,pos,vel,acc):
    
    # Vectors to store parameter values at each step

    ene_kin_aver = np.empty(NSteps)
    ene_pot_aver = np.empty(NSteps)
    temperature  = np.empty(NSteps)
    virial       = np.empty(NSteps)
    pressure     = np.empty(NSteps)
    dist_sq      = np.empty(NSteps)
    
    init_pos     = np.copy(pos)             #Copy the initial Position
    
    final_displacement = np.zeros((N,DIM))  # Store the displacement of each particle at kth step
   


    
    volume=L**3

  
    """
    with nb.objmode(): 
        if os.path.exists('energy2'):
            os.remove('energy2')
        f = open('traj.xyz', 'w')
    """
    for k in range(NSteps):

        # Refold positions according to periodic boundary conditions
        pos=wrap(pos,L)

        # r(t+dt) modify positions according to velocity and acceleration
        displace           = deltat*vel + 0.5*(deltat**2.0)*acc
        pos                = pos + displace                       # Step 1
        final_displacement+= displace*L
        dist_arr           = np.ravel((final_displacement - init_pos))
        dist_sq[k]         = np.dot(dist_arr,dist_arr)             # Sum of displacement squared at each step  

        # Calculate temperature
        ene_kin_aver[k],temperature[k] = Calculate_Temperature(vel,L,DIM,N)

        # Rescale velocities and take half step
        chi = np.sqrt(temp/temperature[k])     # For NVT Ensemble
        #chi  =1                               # For NVE Ensemble
        vel = chi*vel + 0.5*deltat*acc # v(t+dt/2) Step 2

        # Compute forces a(t+dt),ene_pot,virial
        acc, ene_pot_aver[k], virial[k] = LJ_Forces(pos,acc,epsilon,L,DIM,N) # Step 3

        # Complete the velocity step 
        vel = vel + 0.5*deltat*acc # v(t+dt/2) Step 4

        # Calculate temperature
        
        ene_kin_aver[k],temperature[k] = Calculate_Temperature(vel,L,DIM,N)
        
        
        # Calculate pressure
        pressure[k]= density*temperature[k] + virial[k]/volume

        # Print output to file every DumpFreq number of steps
        """
        if(k%DumpFreq==0): # The % symbol is the modulus so if the Step is a whole multiple of DumpFreq then print the values

            f.write("%s\n" %(N)) # Write the number of particles to file
            # Write all of the quantities at this step to the file
            f.write("Energy %s, Temperature %.5f\n" %(ene_kin_aver[k]+ene_pot_aver[k]),temperature[k]))
            for n in range(N): # Write the positions to file
                f.write("X"+" ")
                for l in range(DIM):
                    f.write(str(pos[n][l]*L)+" ")
                f.write("\n")

        #Simple prints without formating are supported
        """
        if (k%10==0):
            print("step: ",k,"KE: ",ene_kin_aver[k],"PE :",ene_pot_aver[k],"\n Total Energy: ",ene_kin_aver[k]+ene_pot_aver[k])
            #print("\rStep: {0} KE: {1}   PE: {2} Energy:  {3}".format(k, ene_kin_aver[k], ene_pot_aver[k],ene_kin_aver[k]+ene_pot_aver[k]))
            #sys.stdout.write("\rStep: {0} KE: {1}   PE: {2} Energy:  {3}".format(k, ene_kin_aver[k], ene_pot_aver[k],ene_kin_aver[k]+ene_pot_aver[k]))
            #sys.stdout.flush()
        
        
    return ene_kin_aver, ene_pot_aver, temperature, pressure, pos, vel,  acc,  dist_sq/N
    



def plot():
    #//////////Initial Position Plot
    plt.figure(num=1,figsize=[16,12])
    plt.title ("Final Position of particles", fontsize=20)
    ax2 = plt.axes(projection='3d')
    ax2.scatter3D(pos[:,0], pos[:,1],pos[:,2], c=pos[:,2], cmap='Greens')
    plt.show()
    #///////
    plt.figure(num=2,figsize=[10,6])
    plt.title("Radial Distribution function", fontsize=20)
    plt.ylabel("g(r)", fontsize=20)
    plt.xlabel("r", fontsize=20)
    plt.plot(grbin,grhist,'-')
    plt.show()
    
    plt.figure(num=3,figsize=[15,25])
    plt.rc('xtick', labelsize=15) 
    plt.rc('ytick', labelsize=15)
    plt.subplot(6, 1, 1)
    plt.plot(ene_kin_aver,'k-')
    plt.ylabel(r"$E_{K}$", fontsize=20)
    plt.subplot(6, 1, 2)
    plt.plot(ene_pot_aver,'k-')
    plt.ylabel(r"$E_{P}$", fontsize=20)
    plt.subplot(6, 1, 3)
    plt.plot(temperature,'k-')
    plt.ylabel(r"$T$", fontsize=20)
    plt.subplot(6, 1, 4)
    plt.plot(pressure,'k-')
    plt.ylabel(r"$P$", fontsize=20)
    plt.subplot(6, 1, 5)
    plt.plot(ene_pot_aver+ene_kin_aver,'k-')
    plt.ylabel(r"$Total Energy$", fontsize=20)
   
    
    
    
    #/////////////////////////
    
    #//////////Mean square displacemment Plot
    plt.figure(num=5,figsize=[10,6])
    plt.title ("Mean square displacement of particles", fontsize=20)
    plt.ylabel("$<r^2>$", fontsize=20)
    plt.xlabel("simulation time", fontsize=20)
    plt.plot(sim_time,dist_meansq,'-')
    plt.show()



@jit(nopython=True,error_model="numpy")
def rdf(pos,grdelta,rmax,L,N):         # Radial distribution function
    # Loop over all pairs of particles
    maxgrbin=int(rmax/grdelta)+1
    grdata=np.zeros(maxgrbin)
    for i in range(N-1):
        for j in range(i+1,N):  #i+1 to N ensures we do not double count
            Sij = pos[i]-pos[j] 
            for l in range(DIM): # Periodic interactions
                if (np.abs(Sij[l])>0.5):
                    Sij[l] = Sij[l] - np.copysign(1.0,Sij[l]) # If distance is greater than 0.5  (scaled units) then subtract 0.5 to find periodic interaction distance.

            Rij   = L*Sij           # Scale the box to the real units in this case reduced LJ units
            Rsqij = np.dot(Rij,Rij) # Calculate the square of the distance

            if(Rsqij < rmax**2):
                seperation=np.sqrt(Rsqij)
                grbin=int(seperation/grdelta)
                grdata[grbin]+=2
    
    for grbin in range(maxgrbin):
        rinner = grbin*grdelta
        router = rinner + grdelta
        shellvol = (4.0*np.pi/3.0)*(router**3 - rinner**3)
        grdata[grbin] = (L**3/(N*(N-1)))*grdata[grbin]/(shellvol)
    grbin=np.arange(maxgrbin)
    grbin=grbin*grdelta
    
    
    return grbin,grdata

    
   
    
        
#------------------------------------------------------    

pos,vel,L         = initialise_config(N,DIM,density,temp)        #Initialise the setup



#//////////Initial Position Plot
plt.figure(num=0,figsize=[16,12])
plt.title ("Initial Position of particles")
ax = plt.axes(projection='3d')
ax.scatter3D(pos[:,0], pos[:,1],pos[:,2], c=pos[:,2], cmap='Greens')
plt.show()
#///////

acc = (np.random.randn(N,DIM)-0.5)

ene_kin_aver, ene_pot_aver, temperature, pressure, pos, vel, acc,dist_meansq = main(NSteps,1,deltat,temp,DumpFreq,epsilon,DIM,N,Rcutoff,pos,vel,acc)
grbin,grhist=rdf(pos,0.1,L/2,L,N)

print ("Diffusion const. calculated from Einsein Relation : ", dist_meansq/(6*deltat*NSteps))
sim_time = [k*deltat for k in range(NSteps)]     #Convert step no. to actual time 


# Plot all of the quantities


plot()