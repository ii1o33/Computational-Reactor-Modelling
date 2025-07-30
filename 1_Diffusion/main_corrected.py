# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:31:55 2025

@author: socce
"""

#This code formulates diffusion equation in neutronics as an eigenvalue/vector problems
#Then solves the problem through power iteration method.

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import linalg
from scipy import integrate
import pandas as pd
import seaborn as sns
import csv
import os
import time

#Constant parameters(user input)
L = 1
ni = 101
D_const = 5.22079e-3        #=from answer sheet(usded for assignment 2 comparison). Originally, 5.22383e-3
Sig_a = 6.7484              #=from answer sheet(usded for assignment 2 comparison). Originally, 6.74752
nu_Sig_f = 9.7933           #=from answer sheet(usded for assignment 2 comparison). Originally, 9.76220
#D_const = 5.22383e-3        
#Sig_a = 6.74752              
#nu_Sig_f = 9.76220
phi_init = 1        #1=constant 1; 2=uniform distribution bet. 0 and 2 at each node.  
keff_init = 1
conv_criteria = 2   #1=Using k; 2=Using S  
normalisation_plot = 2   # 0=No normalisation; 1=Using avg value; 2=L2 norm becomes 1
noramlisation_iter = 0 #=1 if flux is to be normalised at each iterations
fix_iteration = 0    #=1 if the iteration is to be terminated with the fixted number of iterations
n_iteration = 125    #Specify the number of iterations to terminate the calculation

#Switch for plotting (user input)
plotQ2_2 = 0
plotQ2_4 = 0
plotQ2_5 = 0; plotQ2_5_Niter = 0  #=0 if all is to be plotted
plotQ3flux = 0; plotQ3flux_Niter = 0  #=0 if all is to be plotted
plotQ3keff = 0
PlotQ3_multiRun = 0
plotQ4 = 0
plotQ5 = 0

#Switch for data saving (user input)
Q2_kandS= 0
Q3_multiRuns = 0
Q4_diff = 0
Q5_iso = 0

#Constant parameters(calc. auto.)
n_interval = ni-1
x = np.linspace(-L*0.5, L*0.5, ni)
del_x = L/n_interval

#Construct A
A = np.zeros((ni,ni))
A[0,0] = 3*Sig_a*del_x/8 + D_const/del_x + 0.5
A[0,1] = Sig_a*del_x/8 - D_const/del_x
A[ni-1,ni-2] = Sig_a*del_x/8 - D_const/del_x
A[ni-1,ni-1] = 3*Sig_a*del_x/8 + D_const/del_x + 0.5
for k in range(1,n_interval):
    A[k,k-1] = del_x*Sig_a/8 - D_const/del_x
    A[k,k] = 3*del_x*2*Sig_a/8 + 2*D_const/del_x
    A[k,k+1] = del_x*Sig_a/8 - D_const/del_x
    
#Matrices to store flux and k data vursus iteration
phi_mat = []
k_mat = []
    
#Construct initial guess of phi_new
phi_new = np.zeros(ni)
if phi_init == 1:
    phi_new[:] = 1 
elif phi_init == 2:
    for i in range(ni):
        phi_new[i] = np.random.uniform(0.0,2.0)
if noramlisation_iter == 1:
    phi_new = phi_new/np.linalg.norm(phi_new)

#Construct D
D = np.zeros(ni)





#Start timing for main iteration
start_cpu = time.process_time()
start_wall = time.time()
start_io = time.perf_counter()

#Prepare Iterative calculation (0th iteration)
err = 1
k_new = keff_init
if phi_init == 1:
    integ_phi_old = 1
elif phi_init == 2:
    integ_phi_old = integrate.cumulative_trapezoid(phi_new, x, initial=0)[-1]/L
phi_mat.append(phi_new)
k_mat.append(k_new)
iteration = 0

#Iterative calculation
while err >= 0.00001:
    iteration += 1
    
    #Construct D
    D[0] = del_x*nu_Sig_f*(phi_new[0]+phi_new[1])/(4*k_new)
    D[ni-1] = del_x*nu_Sig_f*(phi_new[ni-2]+phi_new[ni-1])/(4*k_new)
    for k in range(1,ni-1):
        D[k] = nu_Sig_f*(phi_new[k-1]+2*phi_new[k]+phi_new[k+1])*del_x/(4*k_new)
    
    #Calc. phi(the main iterative calculation)
    phi_new = linalg.solve(A,D)
    if noramlisation_iter == 1:
        phi_new = phi_new/np.linalg.norm(phi_new)
    #phi_new = phi_new/(integrate.cumulative_trapezoid(phi_new, x, initial=0)[-1]/L)    
    
    #Calc. integral of phi along the skab for normalisation of phi and iterative calculation of k.
    #The integration is performed using trapezoidal rule by assuming linear variation of phi between two consecutive nodes.
    integ_phi_new = integrate.cumulative_trapezoid(phi_new, x, initial=0)[-1]/L
    
    #Calc. k_new (nu_Sig_f is constant so factored out of the integrals and cancelled out each other.)
    k_new = k_new*integ_phi_new/integ_phi_old
    
    #Calc. error
    if conv_criteria == 1:
        err = abs((k_new - k_mat[-1])/k_mat[-1])
    elif conv_criteria == 2:
        #Note nu_Sig_f is factored out and cancelled one another
        err_mat = np.subtract(phi_new/k_new, phi_mat[-1]/k_mat[-1]) 
        err_mat = np.divide(err_mat,phi_mat[-1]/k_mat[-1])
        err = max(abs(err_mat))      
    
    #Terminate iteration if the specified number of iterations is achived
    if fix_iteration == 1:
        if iteration == n_iteration:
            err = 0.0000001
            
    err123 = abs((k_new - k_mat[-1])/k_mat[-1])
    
    #Prep next iter. by saving current data
    integ_phi_old = integ_phi_new
    k_mat.append(k_new)
    phi_mat.append(phi_new)
    
    
    
    



#End timing and print CPU, wall and input/ouput read times    
end_cpu = time.process_time()
cpu_time = end_cpu - start_cpu
print(f"CPU Time: {cpu_time:.6f} seconds")
end_wall = time.time()
wall_time = end_wall - start_wall
print(f"Wall Time: {wall_time:.6f} seconds")
end_io = time.perf_counter()
io_time = end_io - start_io
print(f"I/O Read Time: {io_time:.6f} seconds")


print(iteration)
print(k_mat[-1])
print(err123)
#AAA = np.linalg.eig(A)
#print(AAA[0][0]/AAA[0][1])
#print(k_mat[-1])

#Normalise before plotting
phi_normalised = []
for i in range(len(phi_mat)):
    if normalisation_plot == 1:
        phi_normalised.append(phi_mat[i]/(integrate.cumulative_trapezoid(phi_mat[i], x, initial=0)[-1]/L))
    elif normalisation_plot == 2:
        phi_normalised.append(phi_mat[i]/np.linalg.norm(phi_mat[i]))



#Date export/save for Q2.2
if Q2_kandS == 1:
    fileNameQ5 = "Q2_2.csv"
    if not os.path.exists(fileNameQ5):
        with open(fileNameQ5, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(phi_normalised[-1])  # Write each value in a new row
    else:
        with open(fileNameQ5, "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(phi_normalised[-1])  # Write each value in a new row

#Date export/save for Q3
if Q3_multiRuns == 1:
    if conv_criteria == 1:
        fileNameQ3 = "Q3_multiRun_k.csv"
    elif conv_criteria == 2:
        fileNameQ3 = "Q3_multiRun_S.csv"
    if not os.path.exists(fileNameQ3):
        with open(fileNameQ3, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([iteration])   # Wrap `iteration` in a list
    else:
        with open(fileNameQ3, "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([iteration])   # Wrap `iteration` in a list

#Date export/save for Q4
if Q4_diff == 1:
    #Calc. data first
    y = np.cos(3.07343*x)
    y /= np.linalg.norm(y)
    k_analytical = 1.43628
    k_diff = abs(k_analytical - k_mat[-1])
    flux_diff_mat = np.subtract(y/k_analytical, phi_normalised[-1]/k_mat[-1])
    flux_diff_mat = np.divide(flux_diff_mat,y/k_analytical)
    flux_diff_max = max(abs(flux_diff_mat)) 
    
    if conv_criteria == 1:
        fileNameQ3 = "Q4_diff_k.csv"
    elif conv_criteria == 2:
        fileNameQ3 = "Q4_diff_S.csv"
    if not os.path.exists(fileNameQ3):
        with open(fileNameQ3, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([n_interval,k_diff, flux_diff_max, iteration])   # Wrap `iteration` in a list
    else:
        with open(fileNameQ3, "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([n_interval,k_diff, flux_diff_max, iteration])   # Wrap `iteration` in a list
            
            
#Date export/save for Q5
if Q5_iso == 1:
    fileNameQ5 = "Q5_worst.csv"
    if not os.path.exists(fileNameQ5):
        with open(fileNameQ5, "w", newline='') as file:
            writer = csv.writer(file)
            for value in phi_normalised[-1]:  # Iterate over the data
                writer.writerow([value])  # Write each value in a new row
    else:
        with open(fileNameQ5, "a", newline='') as file:
            writer = csv.writer(file)
            for value in phi_normalised[-1]:  # Iterate over the data
                writer.writerow([value])  # Write each value in a new row



#plot for Q2.2
if plotQ2_2 == 1:
    x = np.linspace(-L*0.5, L*0.5, 101)
    with open('Q2_2.csv', mode='r') as file:  # Open the file in read mode
        reader = csv.reader(file)
        rows = list(reader)  # Read all rows into a list
        
    # Assign the first and second rows to variables
    phi_k = [float(x) for x in rows[0]]  # Convert elements to floats
    phi_S = [float(x) for x in rows[1]]  # Convert elements to floats
    
    #plot
    fig1 = plt.figure(figsize=(12,8))
    ax1=fig1.add_subplot(111)
    ax1.plot(x,phi_k,linestyle='-', linewidth=3)
    ax1.plot(x,phi_S,linestyle='--', linewidth=3)
    ax1.set_xlabel("X",fontsize=16)
    ax1.set_ylabel("Normalised Flux",fontsize=16)
    plt.legend(['Convergence criterion based on k', 'Convergence criterion based on S'],fontsize=16)
    plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    plt.tight_layout()
    plt.show()




#Plot for Q2.4
if plotQ2_4 == 1:
    fig1 = plt.figure(figsize=(12,8))
    ax1=fig1.add_subplot(111)
    ax1.plot(k_mat)
    ax1.set_xlabel("Number of iterations",fontsize=16)
    ax1.set_ylabel("k-eff",fontsize=16)
    plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    # Add a text box
    if conv_criteria == 1:
        textstr = "Convergence criterion based on k\n\nFinal k-eff: 1.43585\nError in k-eff: 9.70930e-06"
    elif conv_criteria == 2:
        textstr = "Convergence criterion based on S\n\nFinal K-eff: 1.43609\nError in K-eff: 7.31854e-08"
    ax1.text(0.2, 0.5, textstr, transform=ax1.transAxes, fontsize=30,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    plt.tight_layout()
    plt.show()



#plot for Q2.5
if plotQ2_5 == 1:
    fig1 = plt.figure(figsize=(12,8))
    ax1=fig1.add_subplot(111)
    if plotQ2_5_Niter == 0:
        for i in range(len(phi_normalised)):
            ax1.plot(x,phi_normalised[i])
    else:         
        for i in range(plotQ2_5_Niter):
            ax1.plot(x,phi_normalised[i])
    ax1.set_xlabel("X",fontsize=23)
    ax1.set_ylabel("Normalised Flux",fontsize=23)
    if plotQ2_5_Niter < 16 and plotQ2_5_Niter != 0:
        plt.legend(['iter=1','iter=2','iter=3','iter=4','iter=5','iter=6','iter=7','iter=8','iter=9','iter=10','iter=11','iter=12','iter=13','iter=14','iter=15'],fontsize=16)
    plt.tick_params(axis='x', labelsize=20)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=20)  # Y-axis tick label font size
    plt.tight_layout()
    plt.show()




#Plot for Q3_keff
if plotQ3keff == 1:
    fig1 = plt.figure(figsize=(12,8))
    ax1=fig1.add_subplot(111)
    ax1.plot(k_mat)
    ax1.set_xlabel("Number of iterations",fontsize=23)
    ax1.set_ylabel("k-eff",fontsize=23)
    plt.tick_params(axis='x', labelsize=20)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=20)  # Y-axis tick label font size
    plt.tight_layout()
    plt.show()
    
    

#plot for Q3_flux
if plotQ3flux == 1:
    fig1 = plt.figure(figsize=(12,8))
    ax1=fig1.add_subplot(111)
    if plotQ3flux_Niter == 0:
        for i in range(len(phi_normalised)):
            ax1.plot(x,phi_normalised[i])
    else:         
        for i in range(plotQ3flux_Niter):
            ax1.plot(x,phi_normalised[i])
    ax1.set_xlabel("X",fontsize=16)
    ax1.set_ylabel("Normalised Flux",fontsize=16)
    if plotQ3flux_Niter == 0:
        plt.legend(['iter=1','iter=2','iter=3','iter=4'],fontsize=16)
    plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    plt.tight_layout()
    plt.show()  
    

#plot for Q3_multiRun_k
if PlotQ3_multiRun == 1:
    # Load the data
    if conv_criteria == 1:
        file_path = 'Q3_multiRun_k.csv'
        reference_value = 64
        Nbins = 34
    elif conv_criteria == 2:
        file_path = 'Q3_multiRun_S.csv'
        reference_value = 149
        Nbins = 188
    data = pd.read_csv(file_path, header=None, names=["Values"])
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data["Values"], kde=False, bins=Nbins, color='blue')
    plt.axvline(x=reference_value, color='red', linestyle='--', linewidth=2, label=f'Reference: Initial flux guess = 1 at all mesh points ({reference_value} iterations)')
    plt.xlabel("Number of iterations", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    plt.tight_layout()
    plt.show()

    
    
    
# Plot for Q4
if plotQ4 == 1:
    # Load the CSV file
    data_file = 'Q4_diff_S.csv'
    data = pd.read_csv(data_file)
    
    # Ensure the columns are named properly; update these if your columns have headers
    data.columns = ['MeshCount', 'ErrorK', 'RelativeErrorSource', 'Iteration']
    
    # Convert 'MeshCount' to float
    xx = data['MeshCount'].astype(float)
    
    # Calculate the theoretical values
    y1 = np.power(xx, -1) * 1e-02  # Correct multiplier
    y2 = np.power(xx, -1) * 1.5e-1  # Correct multiplier

    # Plot 1: Number of Meshes vs Error in k
    plt.figure(figsize=(12, 8))
    plt.plot(xx, data['ErrorK'], marker='o', linestyle='-', color='b', label='Difference_k')
    #plt.plot(xx, y1, color='g', label='1st order spatial scheme (reference)')
    #plt.xscale('log')  # Set x-axis to log scale
    #plt.yscale('log')  # Set y-axis to log scale
    plt.xlabel('Number of meshes', fontsize=16)
    plt.ylabel('Difference_k', fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    #plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Number of Meshes vs Relative Error for the Source
    plt.figure(figsize=(12, 8))
    plt.plot(xx, data['RelativeErrorSource'], marker='s', linestyle='--', color='b', label='Relative_difference_source')
    #plt.plot(xx, y2, color='g', label='1st order spatial scheme (reference)')
    #plt.xscale('log')  # Set x-axis to log scale
    #plt.yscale('log')  # Set y-axis to log scale
    plt.xlabel('Number of meshes', fontsize=16)
    plt.ylabel('Relative_difference_source', fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    #plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()
    
    
    
    
#Plot for Q5
if plotQ5 == 1:
    x = np.linspace(-L*0.5, L*0.5, 100)
    # File paths
    file_worst = 'Q5_worst.csv'
    file_aniso = 'Q5_aniso.csv'
    file_iso = 'Q5_iso.csv'
    
    # Read the CSV files
    worst_data = pd.read_csv(file_worst)
    aniso_data = pd.read_csv(file_aniso)
    iso_data = pd.read_csv(file_iso)
    
    # Plotting each file's data
    plt.figure(figsize=(12, 8))
    
    # Plot each column from the "worst" data
    for column in worst_data.columns:
        plt.plot(x, worst_data[column], label=r'Worst anisotropic case ($\overline{\mu} = \frac{2}{3}$)', linestyle='-.', linewidth=3)

    
    # Plot each column from the "aniso" data
    for column in aniso_data.columns:
        plt.plot(x, aniso_data[column], label='Anisotropic case ($\overline{\mu}$ = 0.17942)', linestyle='--', linewidth=5)
    
    # Plot each column from the "iso" data
    for column in iso_data.columns:
        plt.plot(x, iso_data[column], label='Isotropic case ($\overline{\mu}$ = 0)', linewidth=2.5)
    
    # Customize the plot
    plt.xlabel("X", fontsize=16)
    plt.ylabel("Normalised Flux", fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()