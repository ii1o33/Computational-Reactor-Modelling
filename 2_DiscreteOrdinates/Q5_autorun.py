# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 20:46:32 2025

@author: socce
"""

import os
import numpy as np
import subprocess
import csv

N_runs = 12
L_values = np.linspace(100, 4000, N_runs)  # L here means ni (number of nodes)
L_values = np.round(L_values).astype(int)  # Ensure ni is an integer
file_name_main = "Q5_data.csv"

def modify_main(ni_value):
    """ Modify main.py to update ni dynamically """
    with open("main.py", "r") as file:
        lines = file.readlines()
    
    # Find the line containing 'ni =' and replace it
    for i in range(len(lines)):
        if lines[i].strip().startswith("ni ="):
            lines[i] = f"ni = {ni_value}  # Number of nodes\n"  # Ensure it's an integer
            break
    
    with open("main.py", "w") as file:
        file.writelines(lines)

def automate_runs():
    for ni_value in L_values:
        # Update ni dynamically
        modify_main(ni_value)  
        
        # Run the modified main.py
        os.system("python main.py")  

# Run the automation process
if __name__ == '__main__':
    # Create the CSV file if it doesn't exist
    with open(file_name_main, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ni", "max_rel_diff"])
    
    # Run the automation
    automate_runs()
