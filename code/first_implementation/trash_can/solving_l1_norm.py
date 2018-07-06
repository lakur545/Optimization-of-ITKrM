# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:36:45 2018

@author: Niels
"""
## Defining constraints

import numpy as np
from scipy.optimize import minimize
import time

def objective(x):
    return np.linalg.norm(x,ord=1)                    # min ||x||_1

def ineq_constraint1(x,Y,D,i):
    return -np.linalg.norm(Y[:,i]-D@x,ord=2) + noise  # st. ||y-Dx||_2 < epsilon 

                                                      # constraints are always formulated as X > 0 or X = 0
                                                      # therefore our constrain becomes
                                                      # ||y-Dx||_2 < epsilon
                                                      # -||y-Dx||_2 > -epsilon
                                                      # -||y-Dx||_2 + epsilon > 0


## Generating/importing a test set and dictionary



# similar case to what we do
#####################################
Y_len=16  
Y_samples=64*974

D_atoms=25


Y=np.random.rand(Y_len,Y_samples)*25

D=np.random.rand(Y_len,D_atoms)*10
#####################################
  





# Old test set
#Y=np.array([[1,3,4],[2,4,1]]);      # The set of all your ys, where each y is a coloum
#
#D=np.array([[1,5,7,9],[1,3,9,5]]);  # The dictionary, where is coloum is an atom


## Setting up variables and prelocating storage for minimization


noise=10**-3;                       # This we have to play with, it simply our expected 
                                    # error size in the conversion between x and y

zero_threshold=10**-6;              # How small does a number have to be to be seen as zero

itr=Y.shape[1];                     # simply how many y's do we have

dic_len=D.shape[1];                 # how many atoms

nnz_ar=np.zeros([itr,1]);           # pre locate information regardins sparsness

x_ar=np.zeros([dic_len,itr]);       # pre locate storage for the xs

options1={'maxiter':1001}




## Minimization

for i in range(itr):
    if i==0:                                                                              # Timer start, to make a estimate of run time
        cur_t=time.time()
    print("Current iteration {}/{}".format(i+1,itr))
    x0=100+np.zeros([dic_len,1])                                                         # our first guess

    ineqcon1={'type':'ineq','fun':lambda x: ineq_constraint1(x,Y,D,i)}                   #setting up or constraint
    
    sol = minimize(objective,x0,method='SLSQP',constraints=ineqcon1,options=options1)    # using SciPy minimization solver SLSQP
    
    nonzero=np.sum(np.abs(sol['x'])>zero_threshold)                                      # figuring how many intries are below our zero_threshold
    
    nnz_ar[i]=nonzero                                                                    # saving amount of values below zero threshold
    
    x_ar[:,i]=sol['x']                                                                   # saving the found solution
    
    if i==0:
        elapsed_t=1.07*(time.time()-cur_t)                                               # calculate an estimate of total run time before done
        mins=int((itr*elapsed_t)/60)
        secs=int((itr*elapsed_t)%60)
        print("Time until done estimate:{} minutes {} seconds".format(mins,secs))



## Data processing

indicies=np.abs(x_ar)<zero_threshold                # find all the indices of the values below our zero threshold

x_ar[indicies]=0                                    # setting all the values at the previous found entries to zero

Y_estimate=D@x_ar                                   # calculating the estimate of Y using our dictionary and the sparse X

norm_2_error=np.linalg.norm(Y-Y_estimate,ord=2)     # calculating the the norm 2 error between our Y and Y estimate

print("An X matrix was found with each x vector having sparsety {}\n".format(nnz_ar))

print("'General' sparesity is {} out of {}".format(np.sum(nnz_ar)/itr,dic_len))

print("The general error between Y and the Y estimate using the dictionary is {}".format(norm_2_error))

elapsed_t=time.time()-cur_t                                                      
mins=int((elapsed_t)/60)
secs=int((elapsed_t)%60)
print("Total time elapsed {} minutes {} seconds".format(mins,secs))


