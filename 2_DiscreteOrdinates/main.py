# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:14:02 2025

@author: socce
"""

#Python script written for NE8 Assignment1_Discrete ordinates
#Finds neutron distribution and k for a 1D slab whose properties are the same as the Assignment1
#Vacuum boundary conditions imposed

import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import linalg
from scipy import integrate
from scipy.interpolate import interp1d
import pandas as pd
import seaborn as sns
import csv
import os


#User input
deg = 12         #Degree for GaussLegendre quadrature
L = 1           # Length of 1D slab [m]
phi_init = 1    #Initial guess of scalar flux across the slab
k_init = 1      #Initial guess of the eigenvalue k
ni = 202  # Number of nodes
quad = 1        #1=GL; 2=equal spacing
closure = 1     #1=Diamond-differencing; 2=Step method
conv_criteria = 2   #1=Using k; 2=Using S  
normalisation_plot = 2   # 0=No normalisation; 1=Using avg value; 2=L2 norm becomes 1
cross_sec = 2           #1=Exact(Paul's const); 2=taken from Assignment 1
c = 0           #Ratio of scattering and transport cross sections for Q6
mag_source = 1           #Magnitude of source at the left boundary for Q6


#Switch for plotting (user input)
plotQ1_1 = 0
plotQ1_2 = 0
plotQ2 = 0
plotQ4 = 0
plotQ5 = 1
plotQ6_1 = 0
plotQ6_2 = 0
plotQ6_3 = 0

#Switch for saving data\activate Question specific code (user input)
Q1_save = 0
Q2_save = 0
Q3_save = 0
Q5_save = 0
Q5_save_ref = 0
Q5_save_meshInde = 0        #"Q5_save" has to be 1 as well to use this
Q6_save = 1
Q6_save_data = 0


#Constants input
D = 0.522079*pow(10,-2)
Sigma_a = 6.7484
nu_Sig_f = 9.7633  #nu_bar times fission macroscopic cross-section [neutrons*m^(-1)]
Q2_k_ref = 1.3480737062954302


#Calc. other constants based on input
if cross_sec == 1:
    Sig_tr = 1/(3*0.522079*pow(10,-2))    #Transpost macroscopic cross-section (=total macroscopic cross-section - correction term) [m^(-1)]
    Sig_s = Sig_tr - Sigma_a     #Macroscopic cross-section for scattering [m^(-1)]
elif cross_sec == 2:
    Sig_tr = 63.81014186
    Sig_s = 57.06262186
    nu_Sig_f = 9.76220
n_mesh = ni-1
del_x = L/n_mesh
x_mesh = np.linspace(-L*0.5+del_x*0.5, L*0.5-del_x*0.5, n_mesh)
if quad == 1:
    abscissa, weight = np.polynomial.legendre.leggauss(deg)
elif quad == 2:
    del_mu = 2/deg
    abscissa = np.linspace(-1+del_mu/2,1-del_mu/2,deg)
    weight = np.ones(deg)*del_mu
if Q6_save == 1:
    Sig_s = c*Sig_tr
    

#Initial guess
k_new = k_init
phi_new = np.ones(n_mesh)


#Matrices to store information
phi_mat = []
k_mat = []




#-----------------------------------------------------------------------------------------------------




#Main calculation
#Start timing for main iteration
start_cpu = time.process_time()
start_wall = time.time()
start_io = time.perf_counter()


#Prepare Iterative calculation (0th iteration)
err = 1
phi_mat.append(phi_new.copy())
k_mat.append(k_new)
iteration = 0
if Q5_save == 1:
    tol = 0.00001
else:
    tol = 0.00001
    
#Iterative calculation for phi
while err >= 0.000001:
    iteration += 1
    if Q6_save == 0:
        Q = (nu_Sig_f*phi_mat[-1]/(2*k_mat[-1]) + Sig_s*phi_mat[-1]/2)
    elif Q6_save == 1:
        Q = (Sig_s*phi_mat[-1]/2)
    phi_new[:] = 0
    
    #normal calculation
    if plotQ4 == 0:
        #Calculating for different abscissa (discrete angles)
        for deg_iter in range(len(abscissa)):
            mu = abscissa[deg_iter]
            weight_iter = weight[deg_iter]
            
            psi_node = 0
            #Diamond differencing
            if closure == 1:
                #Calculation procedure for mu > 0
                if mu > 0:  #Note mu is never 0 because even number of abscissa is chosen
                    if Q6_save == 1:
                        psi_node = mag_source
                    for i in range(n_mesh):
                        psi = (del_x*Q[i] + 2*abs(mu)*psi_node)/(del_x*Sig_tr + 2*abs(mu))
                        phi_new[i] += weight_iter*psi
                        psi_node = 2*psi - psi_node
                        
                #Calculation procedure for mu < 0
                else:
                    for i in range(n_mesh-1,-1,-1):
                        psi = (del_x*Q[i] + 2*abs(mu)*psi_node)/(del_x*Sig_tr + 2*abs(mu))
                        phi_new[i] += weight_iter*psi
                        psi_node = 2*psi - psi_node
            #Step method            
            elif closure == 2:
                #Calculation procedure for mu > 0
                if mu > 0:  #Note mu is never 0 because even number of abscissa is chosen
                    for i in range(n_mesh):
                        if Q6_save == 1:
                            psi_node = mag_source
                        psi = (del_x*Q[i] + abs(mu)*psi_node)/(del_x*Sig_tr + abs(mu))
                        phi_new[i] += weight_iter*psi
                        psi_node = psi
                        
                #Calculation procedure for mu < 0
                else:
                    for i in range(n_mesh-1,-1,-1):
                        psi = (del_x*Q[i] + abs(mu)*psi_node)/(del_x*Sig_tr + abs(mu))
                        phi_new[i] += weight_iter*psi
                        psi_node = psi
                
                
    #For Q4, save angular flux
    elif plotQ4 == 1:
        Q4_edge = []
        Q4_centre = []
        #Calculating for different abscissa (discrete angles)
        for deg_iter in range(len(abscissa)):
            mu = abscissa[deg_iter]
            weight_iter = weight[deg_iter]
            
            psi_node = 0
            #Calculation procedure for mu > 0
            if mu > 0:  #Note mu is never 0 because even number of abscissa is chosen
                for i in range(n_mesh):
                    psi = (del_x*Q[i] + 2*abs(mu)*psi_node)/(del_x*Sig_tr + 2*abs(mu))
                    phi_new[i] += weight_iter*psi
                    psi_node = 2*psi - psi_node
                    if i == 1:
                        Q4_edge.append(psi.copy())
                    elif i == (n_mesh - 1)/2:
                        Q4_centre.append(psi.copy())
                    
            #Calculation procedure for mu < 0
            else:
                for i in range(n_mesh-1,-1,-1):
                    psi = (del_x*Q[i] + 2*abs(mu)*psi_node)/(del_x*Sig_tr + 2*abs(mu))
                    phi_new[i] += weight_iter*psi
                    psi_node = 2*psi - psi_node
                    if i == 1:
                        Q4_edge.append(psi.copy())
                    elif i == (n_mesh - 1)/2:
                        Q4_centre.append(psi.copy())
                    
    
    
    #Calculate k
    k_new = np.trapz(phi_new, x_mesh) / np.trapz(phi_mat[-1], x_mesh)*k_new
    
    #Calc. error
    if conv_criteria == 1:
        err = abs((k_new - k_mat[-1])/k_mat[-1])
    elif conv_criteria == 2:
        #Note nu_Sig_f is factored out and cancelled one another
        err_mat = np.subtract(phi_new/k_new, phi_mat[-1]/k_mat[-1]) 
        err_mat = np.divide(err_mat,phi_mat[-1]/k_mat[-1])
        err = max(abs(err_mat)) 
    
    #Save the current iteration data
    phi_mat.append(phi_new.copy())
    k_mat.append(k_new)
    


    

#-----------------------------------------------------------------------------------------------------

    
    
    
    
    
    
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


#Normalise before plotting
if Q6_save == 0:
    phi_normalised = []
    for i in range(len(phi_mat)):
        if normalisation_plot == 1:
            phi_normalised.append(phi_mat[i]/(integrate.cumulative_trapezoid(phi_mat[i], x_mesh, initial=0)[-1]/L))
        elif normalisation_plot == 2:
            phi_normalised.append(phi_mat[i]/np.linalg.norm(phi_mat[i]))
elif Q6_save == 1:
    phi_normalised = phi_mat
        





#-----------------------------------------------------------------------------------------------------




#Date export/save for Q1
if Q1_save == 1:
    fileNameQ5 = f"Q1_flux_S{deg}.csv"
    if not os.path.exists(fileNameQ5):
        with open(fileNameQ5, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(phi_normalised[-1])  # Write each value in a new row
    else:
        with open(fileNameQ5, "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(phi_normalised[-1])  # Write each value in a new row
            
            
# Saving data for Q2
if Q2_save == 1:
    if quad == 1:
        csv_filename = "Q2_SN.csv"
    elif quad == 2:
        csv_filename = "Q2_rec.csv"
    
    # Check if the file exists, if not, create it with headers
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Degree", "k"])
    
    # Save deg and k_new to SN.csv
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([deg, abs(Q2_k_ref - k_new)])
        
        
        
        
#Date export/save for Q3
if Q3_save == 1:
    fileNameQ3 = "Q3_SN_flux.csv"
    # Open the file in "w" mode to overwrite it each time
    with open(fileNameQ3, "w", newline='') as file:
        writer = csv.writer(file)
        # Write phi_normalised[-1] into the file
        for value in phi_normalised[-1]:
            writer.writerow([value])  # Each value on a new row

    
    csv_filename = "Q3_SN_k.csv"
    # Check if the file exists, if not, create it with headers
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
    # Save deg and k_new to SN.csv
        writer.writerow([k_new])
        writer.writerow([iteration])
        
            
            
            
#Date export/save for Q5
if Q5_save == 1:    
    #Read reference flux with ni = 7000
    csv_filename_SN_flux = 'Q5_ref_flux.csv'
    if os.path.exists(csv_filename_SN_flux):
        # Read the CSV file as a 1D array
        df_SN = pd.read_csv(csv_filename_SN_flux, header=None)  # No header in this case
        SN_flux_o = df_SN.iloc[:, 0].to_numpy()  # Convert first column to NumPy array
        
        #Interpolate the current flux data so that it has the same number of nodes as the reference
        L = 1.0
        del_x = L/len(SN_flux_o)
        x_mesh_ref = np.linspace(-L*0.5+del_x*0.5, L*0.5-del_x*0.5, len(SN_flux_o))
        flux = phi_normalised[-1]
        #SN_flux = np.interp(x_mesh,x_mesh_ref,SN_flux_o)
        interp_func = interp1d(x_mesh_ref, SN_flux_o, kind="quadratic", fill_value="extrapolate")
        SN_flux = interp_func(x_mesh)

    err_mat = np.subtract(SN_flux, flux) 
    err_mat = np.divide(err_mat,SN_flux)
    max_rel = max(abs(err_mat))
    print(max_rel)

    
    #Save maximum relative difference to csv file
    csv_filename = "Q5_data.csv"
    # Check if the file exists, if not, create it with headers
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
    # Save deg and k_new to SN.csv
        writer.writerow([ni,max_rel])
    
    
    
    
    
    #check mesh independence to find the reference solution
    if Q5_save_meshInde == 1:
        diff_array = np.subtract(SN_flux, flux)
        RMSD = 0
        for i in range(len(diff_array)):
            RMSD += pow(diff_array[i],2)
        RMSD = np.sqrt(RMSD/len(diff_array))
        print(RMSD)
        
    
#Date export/save for Q5
if Q5_save_ref == 1:
    fileNameQ5 = 'Q5_ref_flux.csv'
    with open(fileNameQ5, "w", newline='') as file:
        writer = csv.writer(file)
        for value in phi_normalised[-1]:
            writer.writerow([value])  # Each value on a new row    
    
    
    
    
#Date export/save for Q6
if Q6_save_data == 1:
    fileNameQ5 = f"Q6_c{c}.csv"
    if not os.path.exists(fileNameQ5):
        with open(fileNameQ5, "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(phi_normalised[-1])  # Write each value in a new row
    else:
        with open(fileNameQ5, "a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(phi_normalised[-1])  # Write each value in a new row
#-----------------------------------------------------------------------------------------------------  
    
    
    
    
    
#plot for Q1
if plotQ1_1 == 1:
    x = np.linspace(-L*0.5, L*0.5, 404)
    with open('Q1_flux_diff.csv', mode='r') as file:  # Open the file in read mode
        reader = csv.reader(file)
        rows = list(reader)  # Read all rows into a list
        
    # Assign the first and second rows to variables
    phi_diff = [float(x) for x in rows[0]]  # Convert elements to floats
    
    #plot
    fig1 = plt.figure(figsize=(11,5))
    ax1=fig1.add_subplot(111)
    #ax1.plot(x_mesh_ref,SN_flux_o,linestyle='-', linewidth=3)
    #ax1.plot(x_mesh_ref,SN_flux,linestyle='--', linewidth=3)
    ax1.plot(x,phi_diff,linestyle='-', linewidth=3)
    ax1.plot(x_mesh,phi_normalised[-1],linestyle='--', linewidth=3)
    ax1.set_xlabel("X",fontsize=16)
    ax1.set_ylabel("Normalised Flux",fontsize=16)
    plt.legend(['Diffusion equation', fr'Discrete ordinates method: $S_{{{deg}}}$'], fontsize=16)
    plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    plt.tight_layout()
    plt.show()
    
    print(k_new)
    print(iteration)
    
    #Calculate the max difference between diffusion and discrete orinates
    a = 0
    for i in range(len(phi_normalised[-1])):
        b = abs(phi_normalised[-1][i] - 0.5*(phi_diff[i] + phi_diff[i+1]))
        if b > a:
            a = b
    a = a/max(phi_normalised[-1])
    print(a)
    
if plotQ1_2 == 1:
    #Plot flux distributions of different degrees of discrete oridinates
    with open('Q1_flux_S2.csv', mode='r') as file:  # Open the file in read mode
        reader = csv.reader(file)
        rows = list(reader)  # Read all rows into a list
    phi_S2 = [float(x) for x in rows[0]]  # Convert elements to floats
    
    with open('Q1_flux_S6.csv', mode='r') as file:  # Open the file in read mode
        reader = csv.reader(file)
        rows = list(reader)  # Read all rows into a list
    phi_S6 = [float(x) for x in rows[0]]  # Convert elements to floats
    
    #plot
    fig2 = plt.figure(figsize=(11,5))
    ax2=fig2.add_subplot(111)
    ax2.plot(x_mesh,phi_S2,linestyle='-', linewidth=3)
    ax2.plot(x_mesh,phi_S6,linestyle='-.', linewidth=3)
    ax2.plot(x_mesh,phi_normalised[-1],linestyle='--', linewidth=3)
    ax2.set_xlabel("X",fontsize=16)
    ax2.set_ylabel("Normalised Flux",fontsize=16)
    plt.legend(['S2', 'S6', 'S12'], fontsize=16)
    plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    plt.tight_layout()
    plt.show()
    
    
    
#plot for Q2
if plotQ2 == 1:
    # Define file names
    csv_filename_SN = "Q2_SN.csv"
    csv_filename_rec = "Q2_rec.csv"
    
    # Initialize arrays
    deg_SN, k_SN, deg_rec, k_rec = [], [], [], []
    
    # Read Q2_SN.csv if it exists
    if os.path.exists(csv_filename_SN):
        df_SN = pd.read_csv(csv_filename_SN)
        deg_SN = df_SN["Degree"].tolist()
        k_SN = df_SN["k"].tolist()
    
    # Read Q2_rec.csv if it exists
    if os.path.exists(csv_filename_rec):
        df_rec = pd.read_csv(csv_filename_rec)
        deg_rec = df_rec["Degree"].tolist()
        k_rec = df_rec["k"].tolist()    
        
    #plot
    fig1 = plt.figure(figsize=(12,8))
    ax1=fig1.add_subplot(111)
    ax1.plot(deg_SN, k_SN, linestyle='-', marker='o', markersize=7, linewidth=3, label='Gauss Legendre')
    ax1.plot(deg_rec, k_rec, linestyle='-', marker='o', markersize=7, linewidth=3, label='Equal spacing')
    ax1.set_xlabel("N",fontsize=16)
    ax1.set_ylabel("Difference in k",fontsize=16)
    ax1.set_yscale("log")
    ax1.minorticks_on()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.grid(True, which='minor',linestyle='--', alpha=0.6)
    plt.legend(fontsize=16)
    plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    plt.tight_layout()
    plt.show()
    
#plot for Q4
if plotQ4 == 1:
    degree = [np.arccos(i) for i in abscissa]
    # Create polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 8))
    ax.plot(degree, Q4_edge, linestyle='-', marker='o', markersize=3, linewidth=2, label=r'$\psi$ at edge')
    ax.plot(degree, Q4_centre, linestyle=':', marker='o', markersize=5, linewidth=2, label=r'$\psi$ at centre')
    ax.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    ax.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    plt.legend(fontsize=16, loc="upper left", bbox_to_anchor=(0.5, 0.3))  # (x, y) coordinates
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    
#plot for Q5
if plotQ5 == 1:
    x = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500]
    y = [5.57e-2,1.31e-2,5.81e-3,3.46e-03,2.36e-3,1.74e-3,1.35e-3,1.09e-3,9.05e-4,7.65e-4,6.58e-4]
    # Create polar plot
    fig2 = plt.figure(figsize=(11,5))
    ax2=fig2.add_subplot(111)
    ax2.plot(x,y,linestyle='-', linewidth=3)
    ax2.set_xlabel("Number of nodes",fontsize=16)
    ax2.set_ylabel("RMSD",fontsize=16)
    plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    plt.tight_layout()
    plt.show()

    
    
#plot for Q6
if plotQ6_1 == 1:
    # File names, corresponding c values, and iterations
    files = {
        "Q6_c0.csv": (0, 2),
        "Q6_c0.5.csv": (0.5, 107),
        "Q6_c0.95.csv": (0.95, 624),
        "Q6_c0.99.csv": (0.99, 1668)
    }
    
    # Define line styles for black and white readability
    line_styles = ['-', '--', '-.', ':']
    
    # Initialize plot
    fig1 = plt.figure(figsize=(15, 7))
    ax1 = fig1.add_subplot(111)
    
    # Loop through each file and plot
    for i, (file, (c_value, iterations)) in enumerate(files.items()):
        try:
            # Read CSV without header assumption
            df = pd.read_csv(file, header=None)
            
            # Generate x-mesh assuming equal spacing
            x_mesh = np.linspace(0, 1, df.shape[1])
            
            # Extract flux values
            phi_normalised = df.iloc[0].values
            
            # Plot with different line styles
            label_format = r'c = {:.4f},   Iterations: {:>6}'
            ax1.plot(x_mesh, phi_normalised, linewidth=3, linestyle=line_styles[i % len(line_styles)], label=label_format.format(c_value, iterations))
        
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    # Labels and formatting
    ax1.set_xlabel("X", fontsize=16)
    ax1.set_ylabel("Flux", fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(axis='x', labelsize=15)  
    plt.tick_params(axis='y', labelsize=15)  
    plt.tight_layout()
    plt.show()
    
    
#plot for Q6
if plotQ6_2 == 1:
    # File names, corresponding c values, and iterations
    files = {
        "Q6_c0.99.csv": (0.99, 1668),
        "Q6_c0.999.csv": (0.999, 5091),
        "Q6_c0.9999.csv": (0.9999, 5762),
        "Q6_c1.csv": (1, 2831)
    }
    
    # Define line styles for black and white readability
    line_styles = ['-', '--', '-.', ':']
    
    # Initialize plot
    fig1 = plt.figure(figsize=(15, 7))
    ax1 = fig1.add_subplot(111)
    
    # Loop through each file and plot
    for i, (file, (c_value, iterations)) in enumerate(files.items()):
        try:
            # Read CSV without header assumption
            df = pd.read_csv(file, header=None)
            
            # Generate x-mesh assuming equal spacing
            x_mesh = np.linspace(0, 1, df.shape[1])
            
            # Extract flux values
            phi_normalised = df.iloc[0].values
            
            label_format = r'c = {:.4f},   Iterations: {:>6}'
            ax1.plot(x_mesh, phi_normalised, linewidth=3, linestyle=line_styles[i % len(line_styles)], label=label_format.format(c_value, iterations))
        
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    # Labels and formatting
    ax1.set_xlabel("X", fontsize=16)
    ax1.set_ylabel("Flux", fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(axis='x', labelsize=15)  
    plt.tick_params(axis='y', labelsize=15)  
    plt.tight_layout()
    plt.show()
    
    
#plot for Q6
if plotQ6_3 == 1:  
    x = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1]
    y = [2,36,50,66,84,107,136,179,247,401,429,463,504,556,624,716,856,1097,1668,2831]
    #plot
    fig1 = plt.figure(figsize=(11,5))
    ax1=fig1.add_subplot(111)
    ax1.plot(x,y,linestyle='-', linewidth=3)
    ax1.set_xlabel("c",fontsize=16)
    ax1.set_ylabel("Number of iterations",fontsize=16)
    plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    plt.tight_layout()
    plt.show()
    
    
