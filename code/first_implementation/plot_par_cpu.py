# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:45:40 2018

@author: sebas plot entries
"""

import numpy as np
import matplotlib.pyplot as plt

overhead_2 = np.asarray([0.9, 1.97, 7.32, 14.37, 23.93, 37.25, 44.5, 56.37])
y_2 = overhead_2

overhead_4 = np.asarray([1.96, 4.23, 15.59, 31.98, 47.87, 63.9, 98.09, 127.1])
y_4 = overhead_4

y = np.asarray([y_2,y_4])

ratio = np.asarray([16.24, 38.09, 53.86, 56.54, 57.66, 58.21, 60.07, 58.33, 60.25])

executionTime_1 = np.asarray([1.84+0.11, 9.09+0.24, 41.01+0.76, 81.38+1.43, 120.76+2.09, 162.4+2.79, 246.04+4.09, 322.78+5.53])
executionTime_4 = np.asarray([2.42+0.11, 6.5+0.24, 25.85+0.76, 52.33+1.43, 78.06+2.09, 104.5+2.79, 159.6+4.09, 207.8+5.53])

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
