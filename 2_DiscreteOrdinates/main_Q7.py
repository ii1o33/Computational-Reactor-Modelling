# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 01:20:19 2025

@author: socce
"""

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
deg = 20               # Number of discrete angles (Gauss-Legendre quadrature)
L = 1.0                 # Length of 1D slab [m]
ni = 5000                # Number of mesh nodes (so n_mesh = ni - 1 cells)
closure = 1             # 1 = diamond differencing, 2 = step differencing
tol = 1.0e-15            # Convergence tolerance
conv_criteria = 2       # 1 = using k, 2 = using flux source difference
normalisation_plot = 0  # 0=none; 1=avg=1; 2=L2 norm=1
plot_fluxes = 1


# Cross-section data for neutron
nu_Sig_f = 9.7633      
Sig_tr_n = 63.8472977  
Sig_s_n  = 57.0988977  
Sigma_a_n = Sig_tr_n - Sig_s_n



# Cross-section data for photon
Sigma_s_g = 80   
Sigma_c_g = 19.9  
Sigma_f_g = 0.1  
Sigma_tr_g = Sigma_s_g + Sigma_c_g + Sigma_f_g  
nu_photon = 2.0


#Calc. of constants
n_mesh = ni - 1
del_x = L / n_mesh
x_mesh = np.linspace(-0.5*L + 0.5*del_x, 0.5*L - 0.5*del_x, n_mesh)
abscissa, weight = np.polynomial.legendre.leggauss(deg)


#Init. guess
k_new = 1.0
phi_n = np.ones(n_mesh)       # Neutron flux guess
phi_g = np.zeros(n_mesh)      # Photon flux guess (start from zero)



#------------------------------------------------------------------------------------------------



#Fuction for diamond differecing sweep across the slab - called in the main loop
def diamond_difference_sweep(Q, Sig_tr, mu, del_x, psi_in_left=0.0, closure_type=1):
    psi_out = np.zeros(n_mesh)
    psi_node = psi_in_left

    # Diamond difference
    for i in range(n_mesh):
        #Note the sweep is from left to right in default. 
        #However, for right to left sweep, Q and psi is fliped when this function is called.
        denom = del_x * Sig_tr + 2.0 * abs(mu)
        psi = (del_x * Q[i] + 2.0 * abs(mu) * psi_node) / denom
        psi_out[i] = psi
        psi_node = 2.0 * psi - psi_node
    return psi_out


def transport_solve(phi_old, phi_other, Sig_tr_group, Sig_s_group, 
                    source_neu_abs=0.0, # e.g. from n absorption => photons
                    source_fission=0.0, # e.g. from fission => group flux
                    k_value=1.0, 
                    closure_type=1):
    # Build the (isotropic) source Q for diamond difference:
    # Q = 1/2 * Sig_s_group * phi_old + 1/(2*k) * (some fission source)
    # plus anything from absorption of the other group, etc.
    Q = 0.5 * Sig_s_group * phi_old  # scattering within the same group

    # Add extra source terms
    #if this group is photons, Q += 0.5 * Sigma_a_n * phi_other
    #If this group is neutrons, Q += 0.5/k_value * (nu_Sig_f * phi_old + nu_photon*Sigma_f_g * phi_other)
    Q += 0.5 * source_neu_abs  #for photon
    Q += 0.5/k_value * source_fission #for neutron

    # Now do the diamond sweeps for all angles:
    phi_new = np.zeros(n_mesh)
    for d_i in range(len(abscissa)):
        mu = abscissa[d_i]
        w = weight[d_i]
        
        #B.C
        psi_in = 0.0
        # Sweep:
        if mu > 0.0:
            psi_cell = diamond_difference_sweep(Q, Sig_tr_group, mu, del_x, 
                                                psi_in_left=psi_in, 
                                                closure_type=closure_type)
        else:
            # Right-to-left sweep: we might define a reversed Q for convenience
            Q_rev = Q[::-1]
            psi_rev = diamond_difference_sweep(Q_rev, Sig_tr_group, mu, del_x, 
                                               psi_in_left=psi_in, 
                                               closure_type=closure_type)
            # Flip back
            psi_cell = psi_rev[::-1]
        # Accumulate cell-average flux
        phi_new += w * psi_cell
    return phi_new



#------------------------------------------------------------------------------------------------



#Main iteration
k_old = k_new
err = 1.0
iteration = 0

phi_n_list = [phi_n.copy()]   # To store iteration history if you want
phi_g_list = [phi_g.copy()]
k_list     = [k_new]

start_cpu = time.process_time()
start_wall = time.time()

while (err > tol):
    iteration += 1

    #Solver for photon
    photon_source_from_neutrons = Sigma_a_n * phi_n
    phi_g_new = transport_solve(
        phi_old=phi_g,
        phi_other=phi_n, 
        Sig_tr_group=Sigma_tr_g,
        Sig_s_group=Sigma_s_g,
        source_neu_abs=photon_source_from_neutrons,  # unscaled
        source_fission=0.0,                          # no photon self-fission
        k_value=k_old,
        closure_type=closure
    )

    #solve for neutron
    source_fission_n = nu_Sig_f * phi_n + nu_photon * Sigma_f_g * phi_g_new
    phi_n_new = transport_solve(
        phi_old=phi_n,
        phi_other=phi_g_new, 
        Sig_tr_group=Sig_tr_n,
        Sig_s_group=Sig_s_n,
        source_neu_abs=0.0,           # no direct "capture -> neutrons" 
        source_fission=source_fission_n,
        k_value=k_old,
        closure_type=closure
    )

    #k updated
    integral_n_old = np.trapz(phi_n, x_mesh)
    integral_n_new = np.trapz(phi_n_new, x_mesh)
    if integral_n_old < 1.0e-14:
        # avoid division by zero
        k_new = 1.0
    else:
        k_new = k_old * (integral_n_new / integral_n_old)

    
    #Error/residual calculation
    if conv_criteria == 1:
        # error based on k
        err = abs(k_new - k_old)/abs(k_old)
    else:
        # error based on flux difference
        eps_n = np.max(np.abs((phi_n_new - phi_n)/(phi_n+1e-14))) #added 1e-14 to avoid dividing by zero
        eps_g = np.max(np.abs((phi_g_new - phi_g)/(phi_g+1e-14)))
        eps_k = abs(k_new - k_old)/(abs(k_old)+1e-14)
        err = max(eps_n, eps_g, eps_k)

    # Update old fluxes & k
    phi_n[:] = phi_n_new
    phi_g[:] = phi_g_new
    k_old    = k_new

    phi_n_list.append(phi_n.copy())
    phi_g_list.append(phi_g.copy())
    k_list.append(k_new)

end_cpu = time.process_time()
cpu_time = end_cpu - start_cpu
print(f"CPU Time: {cpu_time:.6f} s")
end_wall = time.time()
wall_time = end_wall - start_wall
print(f"Wall Time: {wall_time:.6f} s")



#-------------------------------------------------------------------------------------------


#Normalisation
if normalisation_plot == 1:
    # e.g. normalise so average flux is 1
    avg_n = np.trapz(phi_n, x_mesh)/L
    avg_g = np.trapz(phi_g, x_mesh)/L
    phi_n_plot = phi_n / avg_n
    phi_g_plot = phi_g / (avg_g if avg_g>1e-14 else 1.0)
elif normalisation_plot == 2:
    # normalise so L2 norm = 1
    norm_n = np.linalg.norm(phi_n)
    norm_g = np.linalg.norm(phi_g)
    phi_n_plot = phi_n / (norm_n if norm_n>1e-14 else 1.0)
    phi_g_plot = phi_g / (norm_g if norm_g>1e-14 else 1.0)
else:
    phi_n_plot = phi_n
    phi_g_plot = phi_g




#--------------------------------------------------------------------------------------------



#plot
if plot_fluxes:
    fig1 = plt.figure(figsize=(12,8))
    ax1=fig1.add_subplot(111)
    ax1.plot(x_mesh, phi_n_plot, 'o-', label=r'$\phi$ for neutron')
    ax1.plot(x_mesh, phi_g_plot, 'x-', label=r'$\phi$ for photon')
    ax1.set_xlabel("X",fontsize=16)
    ax1.set_ylabel("Flux (not normalised)",fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    plt.tight_layout()
    plt.show()




print(k_new)