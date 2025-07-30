# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:46:32 2025

@author: socce
"""
import os
import numpy as np
import subprocess
import pandas as pd
import seaborn as sns
import csv


N_runs = 45
L_values = np.linspace(0.05, 5, N_runs)  # Generate 2000 values of L
file_name_main = 'Q3_main.csv'

def modify_main(L_value):
    """ Modify main.py to update L dynamically """
    with open("main.py", "r") as file:
        lines = file.readlines()
    
    # Find the line containing 'L =' and replace it
    for i in range(len(lines)):
        if lines[i].strip().startswith("L ="):
            lines[i] = f"L = {L_value:.6f}  # Length of 1D slab [m]\n"
            break
    
    with open("main.py", "w") as file:
        file.writelines(lines)
        
        
def modify_diffusionSolver(L_value):
    """ Modify diffusion_solver.py to update L dynamically """
    with open("diffusion_solver.py", "r") as file:
        lines = file.readlines()
    
    # Find the line containing 'L =' and replace it
    for i in range(len(lines)):
        if lines[i].strip().startswith("L ="):
            lines[i] = f"L = {L_value:.6f}  # Length of 1D slab [m]\n"
            break
    
    with open("diffusion_solver.py", "w") as file:
        file.writelines(lines)



def automate_runs():
    for L in L_values:
        #Update length of the slab
        modify_main(L)  # Update L value in main.py
        modify_diffusionSolver(L)  # Update L value in diffusion_solver.py
        
        
        #Run each solver
        os.system("python diffusion_solver.py")  # Run diffusion_solver.py
        os.system("python main.py")  # Run main.py
        
        
        #Read the output data from each solver
        csv_filename_SN_flux = 'Q3_SN_flux.csv'
        if os.path.exists(csv_filename_SN_flux):
            # Read the CSV file as a 1D array
            df_SN = pd.read_csv(csv_filename_SN_flux, header=None)  # No header in this case
            SN_flux = df_SN.iloc[:, 0].to_numpy()  # Convert first column to NumPy array
            
        csv_filename_diff_flux = 'Q3_diffusion_flux.csv'
        if os.path.exists(csv_filename_diff_flux):
            # Read the CSV file as a 1D array
            df_SN = pd.read_csv(csv_filename_diff_flux, header=None)  # No header in this case
            diff_flux = df_SN.iloc[:, 0].to_numpy()  # Convert first column to NumPy array
            
        csv_filename_diffusion_k = 'Q3_diffusion_k.csv'
        if os.path.exists(csv_filename_diffusion_k):
            # Read the CSV file as a DataFrame
            df_k = pd.read_csv(csv_filename_diffusion_k, header=None)  # No header in this case
            # Extract the scalar value
            diffusion_k = df_k.iloc[0, 0]  # Extract first value as a scalar
            diffusion_iter = df_k.iloc[1, 0]
            
        csv_filename_SN_k = 'Q3_SN_k.csv'
        if os.path.exists(csv_filename_SN_k):
            # Read the CSV file as a DataFrame
            df_k = pd.read_csv(csv_filename_SN_k, header=None)  # No header in this case
            # Extract the scalar value
            SN_k = df_k.iloc[0, 0]  # Extract first value as a scalar
            SN_iter = df_k.iloc[1, 0]
            
        
        #Calculation for max relative difference of flux
        #max_rel = 0
        #for i in range(len(SN_flux)):
        #    b = abs(SN_flux[i] - 0.5*(diff_flux[i] + diff_flux[i+1]))
        #    if b > max_rel:
        #        max_rel = b
        #max_rel = max_rel/max(SN_flux)
        flux = np.zeros(len(SN_flux))
        for i in range(len(SN_flux)):
            flux[i] = 0.5*(diff_flux[i] + diff_flux[i+1])
        err_mat = np.subtract(SN_flux, flux) 
        err_mat = np.divide(err_mat,SN_flux)
        max_rel = max(abs(err_mat))
        
        
        #Calc. for difference in k
        diff_k = abs(diffusion_k - SN_k)
        
        
        #Save the data for each length as a separate row in csv file.
        with open(file_name_main, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([L,diff_k,max_rel,diffusion_iter,SN_iter])

    
            
        



# Run the automation process
if __name__ == '__main__':
    if not os.path.exists(file_name_main):
        with open(file_name_main, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["L", "difference_k", "rel_diff_flux", "iter_diffusion", "iter_SN"])
    automate_runs()