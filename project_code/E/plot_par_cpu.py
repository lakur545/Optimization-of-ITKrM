# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:45:40 2018

@author: sebas plot entries
"""

import numpy as np
import matplotlib.pyplot as plt

overhead_2 = np.asarray([0.15455, 0.2772, 0.3455, 0.66485, 0.91115, 0.94863, 1.0795, 2.3418])
y_2 = overhead_2

overhead_4 = np.asarray([0.066775, 0.8842, 2.04155, 3.746475, 5.446525, 7.34185, 11.2488, 12.9513])
y_4 = overhead_4

y = np.asarray([y_2,y_4])

ratio = np.asarray([3.8445, 12.1862, 16.2277, 16.8752, 16.8622, 16.5196, 16.3668, 16.4651, 17.696])

executionTime_1 = np.asarray([0.5233, 3.1247, 14.4459, 28.3679, 42.4999, 58.5327, 84.4683, 112.4648])
executionTime_4 = np.asarray([0.7509, 1.752, 5.8592, 10.6757, 17.0929, 21.8951, 35.2517, 45.5516])

nTrainingData = np.asarray([10,100, 500, 1000, 1500, 2000, 3000, 4000])
x = nTrainingData

nTrainingData_2 = np.asarray([10,100, 500, 1000, 1500, 2000, 3000, 4000, 6000])

#plt.figure('Overhead')
#plt.plot(x,y_2,'-o') # 'o' makes the dots
#plt.plot(x,y_4,'-o') # 'o' makes the dots
#plt.xlabel('nTrainingData', fontsize=18)
#plt.ylabel('Overhead [s]', fontsize=16)
#plt.gca().legend(('2 threads','4 threads'))



#plt.figure('Ratio')
#plt.plot(nTrainingData_2,ratio,'-o') # 'o' makes the dots
#plt.xlabel('nTrainingData', fontsize=18)
#plt.ylabel('ratio', fontsize=16)
#plt.ylim(0, 65)


plt.figure(1, dpi=200)
plt.plot(x,y_2,'-o',linewidth=2.5, label = '2 processes') # 'o' makes the dots
plt.plot(x,y_4,'-o',linewidth=2.5, label = '4 processes') # 'o' makes the dots
plt.xlabel('nTrainingData', fontsize=16)
plt.ylabel('$T_{oh}$ [s]', fontsize=16)
plt.legend()
ax = plt.gca()
plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
#plt.ylim([-0.5, 0.1])
#plt.xlim([0, 600])
plt.grid(which="major",ls="-", color='grey')
plt.tight_layout()


plt.figure(2, dpi=200)
plt.plot(nTrainingData_2,ratio,'-o',linewidth=2.5) # 'o' makes the dots
plt.xlabel('nTrainingData', fontsize=16)
plt.ylabel('$t_{ratio}$', fontsize=16)
ax = plt.gca()
plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
#plt.ylim([-0.5, 0.1])
#plt.xlim([0, 600])
plt.grid(which="major",ls="-", color='grey')
plt.tight_layout()


plt.figure(3, dpi=200)
plt.plot(x,executionTime_1,'-o',linewidth=2.5, label = '1 process') # 'o' makes the dots
plt.plot(x,executionTime_4,'-o',linewidth=2.5, label = '4 processes') # 'o' makes the dots
plt.xlabel('nTrainingData', fontsize=16)
plt.ylabel('execution time [s]', fontsize=16)
ax = plt.gca()
plt.legend()
plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
#plt.ylim([-0.5, 0.1])
#plt.xlim([0, 600])
plt.grid(which="major",ls="-", color='grey')
plt.tight_layout()
