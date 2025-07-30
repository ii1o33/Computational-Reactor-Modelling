# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 14:36:38 2025

@author: socce
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the tridiagonal matrix A
def create_tridiagonal_matrix(n, main_diag, off_diag):
    main_diagonal = main_diag * np.ones(n)
    off_diagonals = off_diag * np.ones(n - 1)
    A = (
        np.diag(main_diagonal) + 
        np.diag(off_diagonals, k=1) + 
        np.diag(off_diagonals, k=-1)
    )
    return A

# Matrix size
n = 101
x = np.linspace(-0.5, 0.5, 101)

# Parameters for the matrix
main_diag = 1.04769
off_diag = -0.513949

# Create the matrix
A = create_tridiagonal_matrix(n, main_diag, off_diag)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(A)

# Sort eigenvalues and eigenvectors in ascending order of eigenvalues
sorted_indices = np.argsort(eigenvalues)
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Plot the first four dominant eigenvectors
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, sharey=True)
for i in range(4):
    axes[i].plot(x, eigenvectors[:, i], label=f"Eigenvector {i+1}")
    axes[i].legend(fontsize=16)
    axes[i].grid()
    axes[i].tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
fig.text(0.5, 0.04, 'X', ha='center', fontsize=16)
fig.text(0.04, 0.5, 'Normalised Eigenvector', va='center', rotation='vertical', fontsize=16)
plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.show()



