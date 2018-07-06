# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 13:49:06 2018

@author: Niels
"""

import numpy as np
import matplotlib.pyplot as plt

data={}
n=np.array([10,20,40,80,160,320,640,1280])
data["data_10"] = [2455, 2394, 2328, 2359, 2343, 2372, 2347, 2375, 2356, 2343, 2362, 2359, 2359, 2390, 2375, 2328, 2359, 2422, 2424, 2343]
data["data_20"] = [4843, 4736, 4697, 4959, 5777, 6286, 5831, 5888, 5203, 5063, 5084, 5808, 5285, 7063, 5609, 5721, 5685, 4687, 4719, 4638]
data["data_40"] = [10016, 10086, 14296, 10943, 9358, 9938, 10696, 9285, 10261, 10200, 10108, 10103, 11808, 11854, 10391, 9771, 9613, 10110, 9676, 9747]
data["data_80"] = [20334, 19957, 18775, 20048, 19760, 20143, 18987, 18766, 25878, 20630]
data["data_160"] = [36503, 34365, 37549, 46197, 36122, 35761, 34446, 35714, 35692, 35689, 34370, 34144, 35320, 34763, 34421, 33978, 35270, 35280, 34215, 35707]
data["data_320"] = [71128, 72296, 72378, 71534, 70927, 71811, 72006, 71637, 71942, 72195, 71901, 71875, 74875, 89160, 102063, 76443, 76522, 77227, 77663, 70611]
data["data_640"] = [162913, 166917, 154278, 148997, 150407, 152973, 173496, 158094, 156321, 156901]
data["data_1280"] = [298522, 273215, 269974]

mean_data=[]
for key in data.keys():
   mean_data += [np.mean(data[key])]
plt.close('all') 

plt.title("Execution time vs training data amount")


A = np.vstack([n, np.ones(len(n))]).T
m, c = np.linalg.lstsq(A, mean_data)[0]

plt.plot(n, m*n + c, 'r', label='Fitted line')
plt.plot(n,mean_data,'o',label="Meaned times")
plt.legend()
plt.xlabel("Amount of pictures")
plt.ylabel("Execution time [ms]")



### save figure
#plt.tight_layout()
#plt.savefig('pictures_vs_time.eps')