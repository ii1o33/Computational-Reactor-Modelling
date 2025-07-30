# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 15:52:14 2025

@author: socce
"""

import pandas as pd
import matplotlib.pyplot as plt

# Define file path
file_path_main = "Q3_main.csv"  # Update with the correct file path if needed

# Read the CSV file into a DataFrame
df_main = pd.read_csv(file_path_main)

# Extract relevant columns for plotting
L = df_main["L"]

# First graph: difference_k vs rel_diff_flux
fig2 = plt.figure(figsize=(12,8))
ax2=fig2.add_subplot(111)
ax2.plot(L, df_main["difference_k"], label="Difference in k", marker='o', linestyle='-')
ax2.plot(L, df_main["rel_diff_flux"], label="Maximum relative difference of flux", marker='s', linestyle='--')
ax2.set_xlabel("Length [m]",fontsize=16)
#ax2.set_ylabel("Normalised Flux",fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
plt.tight_layout()
plt.grid(True)
plt.show()




# Second graph: iter_diffusion vs iter_SN
fig1 = plt.figure(figsize=(12,8))
ax1=fig1.add_subplot(111)
ax1.plot(L, df_main["iter_diffusion"], label="Diffusion solver", marker='o', linestyle='-')
ax1.plot(L, df_main["iter_SN"], label=r"Discrete Ordinates $S_{12}$", marker='s', linestyle='--')
ax1.set_xlabel("Length [m]",fontsize=16)
ax1.set_ylabel("Number of iterations for convergence",fontsize=16)
plt.legend(fontsize=16)
plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
plt.tight_layout()
plt.grid(True)
plt.show()

