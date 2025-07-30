# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 16:32:39 2025

@author: socce
"""

#This code formulates diffusion equation in neutronics as an eigenvalue/vector problems
#Then solves the problem through power iteration method.

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy import integrate
import pandas as pd
import seaborn as sns
import csv
import os
from matplotlib.ticker import ScalarFormatter

#Constant parameters(user input)
L = 1
ni = 1001
phi_init = 1        #1=constant 1; 2=uniform distribution bet. 0 and 2 at each node.  
keff_init = 1
conv_criteria = 1   #1=Using k; 2=Using S  
noramlisation_iter = 1 #=1 if flux is to be normalised at each iterations
fix_iteration = 1    #=1 if the iteration is to be terminated with the fixted number of iterations
n_iteration = 1000    #Specify the number of iterations to terminate the calculation
power = 100e10

#Switch for plotting (user input)


#Constant parameters(calc. auto.)
n_interval = ni-1
x = np.linspace(-L*0.5, L*0.5, ni)
del_x = L/n_interval

    
#Matrices to store flux and k data vursus iteration
phi_mat = []
k_mat = []

#Construct empty matrices and constants to be updated each iteration
phi_avg = np.zeros(ni-1)
Xe = np.zeros(ni-1)
Sm = 6.1285e22
Sig_t = np.zeros(ni-1)
Sig_s = np.zeros(ni-1)
mu = np.zeros(ni-1)
D = np.zeros(ni-1)
Sig_a = np.zeros(ni-1)
A = np.zeros((ni,ni))
RHS = np.zeros(ni)
nu_Sig_f = 9.76220

#Construct initial guess of phi_new
phi_new = np.zeros(ni)
if phi_init == 1:
    phi_new[:] = power*7.991871865701142e+11 
elif phi_init == 2:
    for i in range(ni):
        phi_new[i] = np.random.uniform(0.0,2.0)








#Prepare Iterative calculation (0th iteration)
err = 1
k_new = keff_init
if phi_init == 1:
    integ_phi_old = power*7.991871865701142e+11
elif phi_init == 2:
    integ_phi_old = integrate.cumulative_trapezoid(phi_new, x, initial=0)[-1]/L
phi_mat.append(phi_new)
k_mat.append(k_new)
iteration = 0

#Iterative calculation
while err >= 0.00001:
    iteration += 1
    
    for i in range(ni-1):
        phi_avg[i] = (phi_new[i]+phi_new[i+1])*0.5
        Xe[i] = 0.2499*phi_avg[i]/(2.116e-5 + 2.2e-23*phi_avg[i])
        Sig_t[i] = 76.2870 + 2.20007e-28*Xe[i] + 7208.66e-28*Sm
        Sig_s[i] = 69.5398 + 6.9934e-28*Xe[i] + 8.6577e-28*Sm
        mu[i] = (12.4768 + 3.48585e-30*Xe[i] + 3.90943e-30*Sm)/Sig_s[i]
        D[i] = 1/(3*(Sig_t[i] - mu[i]*Sig_s[i]))
        Sig_a[i] = 6.74752 + 2.2e-23*Xe[i] + 7.2e-23*Sm
    
    #Construct A
    A[0,0] = 3*Sig_a[0]*del_x/8 + D[0]/del_x + 0.5
    A[0,1] = Sig_a[0]*del_x/8 - D[0]/del_x
    A[ni-1,ni-2] = Sig_a[ni-2]*del_x/8 - D[ni-2]/del_x
    A[ni-1,ni-1] = 3*Sig_a[ni-2]*del_x/8 + D[ni-2]/del_x + 0.5
    for i in range(1,ni-1):
        A[i,i-1] = del_x*Sig_a[i-1]/8 - D[i-1]/del_x
        A[i,i] = 3*del_x*(Sig_a[i-1] + Sig_a[i])/8 + (D[i] + D[i-1])/del_x
        A[i,i+1] = del_x*Sig_a[i]/8 - D[i]/del_x
        
    #Construct RHS
    RHS[0] = del_x*nu_Sig_f*(phi_new[0]+phi_new[1])/(4*k_new)
    RHS[ni-1] = del_x*nu_Sig_f*(phi_new[ni-2]+phi_new[ni-1])/(4*k_new)
    for k in range(1,ni-1):
        RHS[k] = nu_Sig_f*(phi_new[k-1]+2*phi_new[k]+phi_new[k+1])*del_x/(4*k_new)
        
    #Calc. phi(the main iterative calculation)
    phi_new = linalg.solve(A,RHS)
    
    #Calc. integral of phi along the skab for normalisation of phi and iterative calculation of k.
    #The integration is performed using trapezoidal rule by assuming linear variation of phi between two consecutive nodes.
    integ_phi_new = integrate.cumulative_trapezoid(phi_new, x, initial=0)[-1]/L
    
    
    
    #Calc. k_new (nu_Sig_f is constant so factored out of the integrals and cancelled out each other.)
    k_new = k_new*integ_phi_new/integ_phi_old
    
    
    if noramlisation_iter == 1:
        curr_pow = 200*pow(10,6)*1.6021892*pow(10,-19)*3.90488*(integrate.cumulative_trapezoid(phi_new, x, initial=0)[-1]/100)  #Current power[W/cm]
        phi_new = phi_new*power/curr_pow    #Normalise so that the reactor power is 100kW/cm
        
    
    #Calc. error
    if conv_criteria == 1:
        err = abs((k_new - k_mat[-1])/k_mat[-1])
    elif conv_criteria == 2:
        #Note nu_Sig_f is factored out and cancelled one another
        #err_mat = np.subtract(phi_new, phi_mat[-1]) 
        #err_mat = np.divide(err_mat,phi_mat[-1])
        err_mat = np.subtract(phi_new/k_new, phi_mat[-1]/k_mat[-1]) 
        err_mat = np.divide(err_mat,phi_mat[-1]/k_mat[-1])
        err = max(abs(err_mat))      
    
    #Terminate iteration if the specified number of iterations is achived
    if fix_iteration == 1:
        if iteration == n_iteration:
            err = 0.0000001
            
    #err123 = abs((k_new - k_mat[-1])/k_mat[-1])
    
    #Prep next iter. by saving current data
    integ_phi_old = integ_phi_new
    k_mat.append(k_new)
    phi_mat.append(phi_new)
            
        
    print(iteration)
    print(err)
        
        
print(k_mat[-1])
print(k_mat[-2])    
        
        
        
        
        
        
phi_normalised = phi_mat 
       
plot1 = 0
if plot1 == 1: 
    #plot
    fig1 = plt.figure(figsize=(12,8))
    ax1=fig1.add_subplot(111)
    ax1.plot(x,phi_normalised[-1])
    ax1.set_xlabel("X",fontsize=16)
    ax1.set_ylabel("Normalised Flux",fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    plt.tight_layout()
    plt.show()

plot2 = 1
if plot2 == 1:
    xx = np.linspace(-0.5+del_x, 0.5-del_x, ni-1)

    # Plot 1
    fig2 = plt.figure(figsize=(12, 8))
    ax2 = fig2.add_subplot(111)
    ax2.plot(xx, Xe, label=r'$Xe_{o}$')
    plt.xlabel("X", fontsize=16)
    plt.ylabel("Normalised Flux", fontsize=16)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax2.yaxis.get_offset_text().set_fontsize(15)  # Increase offset font size
    plt.legend(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.tight_layout()
    plt.show()

    # Plot 2
    fig3 = plt.figure(figsize=(12, 8))
    ax3 = fig3.add_subplot(111)
    Sm_mat = np.zeros(ni-1)
    Sm_mat[:] = Sm
    ax3.plot(xx, Sm_mat, label=r'$Sm_{o}$')
    plt.xlabel("X", fontsize=16)
    plt.ylabel("Normalised Flux", fontsize=16)
    ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax3.yaxis.get_offset_text().set_fontsize(15)  # Increase offset font size
    plt.legend(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.tight_layout()
    plt.show()
    
    

plotQ3flux=0
#plot for Q3_flux
if plotQ3flux == 1:
    fig1 = plt.figure(figsize=(12,8))
    ax1=fig1.add_subplot(111)  
    for i in range(len(phi_mat)):
        ax1.plot(x,phi_normalised[i], label=f'{i} iterations')
    ax1.set_xlabel("X",fontsize=16)
    ax1.set_ylabel("Normalised Flux",fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    plt.tight_layout()
    plt.show()  
    
plotQ2_2 = 1
#plot for Q2.2
if plotQ2_2 == 1:
    xx = np.linspace(-L*0.5, L*0.5, 101)
    with open('Q2_2.csv', mode='r') as file:  # Open the file in read mode
        reader = csv.reader(file)
        rows = list(reader)  # Read all rows into a list
        
    # Assign the first and second rows to variables
    phi_k = [float(x) for x in rows[0]]  # Convert elements to floats
    phi_S = [float(x) for x in rows[1]]  # Convert elements to floats
    
    phi_new = phi_new/np.linalg.norm(phi_new)
    #phi_S = np.array(phi_S, dtype=np.float64)
    #phi_S = phi_S*max(phi_new)/max(phi_S)
    #plot
    fig1 = plt.figure(figsize=(12,8))
    ax1=fig1.add_subplot(111)
    ax1.plot(x,phi_new,linestyle='-', linewidth=3)
    ax1.plot(xx,phi_S,linestyle='--', linewidth=3)
    ax1.set_xlabel("X",fontsize=16)
    ax1.set_ylabel("Normalised Flux",fontsize=16)
    plt.legend(['With Xe and Sm present', 'Without Xe and Sm present'],fontsize=16)
    plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    plt.tight_layout()
    plt.show()
    
    