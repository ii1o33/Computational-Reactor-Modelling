# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:55:04 2025

@author: socce
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV files
file_diamond = "Q5_diamond.csv"
file_step = "Q5_step.csv"

df_diamond = pd.read_csv(file_diamond)
df_step = pd.read_csv(file_step)


# Define reference line for first-order spatial scheme
x = np.array([df_step['ni'].min(), 800])  # Match ni range
y = 5 * (x / x[0])**-1  # First-order slope (-1 in log-log scale)

# Plot max_rel_diff vs ni for both datasets using log-log scale
plt.figure(figsize=(12,8))
plt.loglog(df_diamond['ni'], df_diamond['max_rel_diff'], marker='o', linestyle='-', label='Diamond difference')
plt.loglog(df_step['ni'], df_step['max_rel_diff'], marker='s', linestyle='--', label='Step method')
plt.loglog(x,y,label='First order spatial scheme',linestyle=':')

# Labels and title
plt.xlabel('Number of nodes',fontsize=16)
plt.ylabel('Maximum relative flux error',fontsize=16)
plt.legend(fontsize=16)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Show the plot
plt.tight_layout()
plt.show()

