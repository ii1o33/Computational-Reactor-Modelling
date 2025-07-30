# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 23:37:46 2025

@author: socce
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy import integrate
import pandas as pd
import seaborn as sns
import csv
import os
from matplotlib.ticker import ScalarFormatter


x = [100e0,100e1,100e2,100e3,100e4,100e5,100e6,100e7,100e8,100e9,100e10]
y = [0.85702,0.85700,0.85684,0.855386,0.84804,0.84032,0.83872, 0.83855,0.83853,0.83853,0.83853]


plot1 = 1
if plot1 == 1: 
    #plot
    fig1 = plt.figure(figsize=(12,8))
    ax1=fig1.add_subplot(111)
    ax1.plot(x,y)
    ax1.set_xlabel("X",fontsize=16)
    ax1.set_ylabel("Normalised Flux",fontsize=16)
    plt.legend(fontsize=16)
    plt.tick_params(axis='x', labelsize=15)  # X-axis tick label font size
    plt.tick_params(axis='y', labelsize=15)  # Y-axis tick label font size
    plt.tight_layout()
    plt.show()