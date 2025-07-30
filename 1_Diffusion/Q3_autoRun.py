import os
import numpy as np
import subprocess
import csv

N_runs = 2000

def automate_runs():
    for i in range(N_runs):
        # Run the Fortran solver and capture output
        os.system("python main_corrected.py")

# Run the automation process
if __name__ == '__main__':
    automate_runs()