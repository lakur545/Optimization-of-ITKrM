# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:24:19 2018

@author: Niels
"""

import numpy as np
import matplotlib.pyplot as plt


n=np.array([1,2,4,8,16,32,64,128])
pictures=1
picture_size=32*32
amount_of_training_data=pictures*(picture_size/(n**2))

plt.close('all') 

plt.title("Window size vs amount of training data it generates")


plt.plot(n,amount_of_training_data)

plt.xlabel("Window size [N x N]")
plt.ylabel(r"Multiplier of training data amount")


### save figure
#plt.tight_layout()
#plt.savefig('window_vs_training_generated.eps')