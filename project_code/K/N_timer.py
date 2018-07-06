# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:52:11 2018

@author: Niels
"""

def Timer(iteration_variable,itr=1,t=[0],itr_show=1):
    import time
    if itr_show==1:    
        print('\rIteration {} of {}'.format(iteration_variable+1, itr), end='') 
    if iteration_variable==1:
        itr=itr-1
        timed=time.time()-t[0]                                           # calculate an estimate of total run time before done
        mins=int((itr*timed)/60)
        secs=int((itr*timed)%60)
        print("\rEstimate until done: {} minutes {} seconds".format(mins,secs))
        t[0]=0
    elif iteration_variable==0 and itr!=1: 
        t[0]+=time.time()
    return

def cont_timer(start_time,stop,printing=0):
    import time
    if stop==1:
        timed=time.time()-start_time                                           # calculate an estimate of total run time before done
        if printing==1:
            print("\n")
            print("{} ms".format(int(timed*1000)))
        return int(timed*1000)
    if stop==0:
        return time.time()
    
def log(data,file_name="output.txt",log_s='0',open_file=0):
    if open_file==1:
        with open(file_name, 'a') as f:
            f.write("\n")
            f.write(log_s)
            f.write("\n")
            f.write("[")
    if open_file==-1:
        with open(file_name, 'a') as f:
            f.write(str(data))
            f.write("]")
            f.write('\n')
            f.close()
    if open_file==0:
        with open(file_name, 'a') as f:
            f.write(str(data))
            f.write(", ") 