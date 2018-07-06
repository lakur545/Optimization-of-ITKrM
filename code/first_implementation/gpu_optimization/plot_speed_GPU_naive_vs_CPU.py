#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:27:20 2018

@author: niels
"""

import numpy as np
import matplotlib.pyplot as plt


my_data = np.genfromtxt('Execution time.csv', delimiter=',')
my_data = my_data[1:-1,:]
n_training_data = my_data[:, 0]
GPU_time = my_data[:, 1]
CPU_time = my_data[:, 2]

plt.close("all")
plt.figure(1, dpi=200)

plt.plot(n_training_data, GPU_time, label = "GPU", linewidth=2.5)
plt.plot(n_training_data, CPU_time, label = "CPU", linewidth=2.5)

plt.xlabel('Amount of training data [N]', fontsize=16)
plt.ylabel('Execution time [s]', fontsize=16)
ax = plt.gca()
plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
plt.grid(which="major",ls="-", color='grey')
plt.legend()
plt.tight_layout()
plt.show()