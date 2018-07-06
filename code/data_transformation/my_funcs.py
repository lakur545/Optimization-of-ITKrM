# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 09:27:42 2018

@author: Niels
"""

def filePath(Title='',FileTypeDescp='',Filetypes='',startdir=''):
    from tkinter import Tk                         ##
    from tkinter.filedialog import askopenfilename ## These are needed for general file dialog 
    root = Tk()                                    ## Setup at tinker window
    ftypes = [(FileTypeDescp,Filetypes)]           ## Variale for handling info regarding what filetypes, and a description of them for the GUI
    ttl  = Title                                   ## Set the title of the GUI window
    dir1 = startdir                                ## Set the starting directory for the GUI
    file_path = askopenfilename(filetypes = ftypes, initialdir = dir1, title = ttl) 
    root.withdraw()                                ## Close the windows
    return file_path                               ## Returns the path to the file


def unpickle(file):
    import pickle                                  ## Standart file I/O from pthon  
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')   ## Using the pickle load the info from the file, which is encoded as bytes
    return dict

def grayVec(vector,pictureDimH,pictureDimW):
    import numpy as np
    size=pictureDimH*pictureDimW                   ## calculate what the full vector length should be, per color layer
    gray=np.floor(0.299*vector[0:size]+0.587*vector[size:2*size]+0.114*vector[2*size:3*size])
    ## The numbers infront of each is the color weight for getting RGB to gray scale, via summing each color layer.
    return gray



def RGB_blocs_vector(data,N):
    import numpy as np
    amount_of_picture_layers=len(data)//3
    newSet = np.zeros((amount_of_picture_layers, N*N*3))
    for i in range(amount_of_picture_layers):
        newSet[i,:]=np.append(np.append(data[i,0:N*N],data[amount_of_picture_layers+i,0:N*N]),data[amount_of_picture_layers*2+i,0:N*N]).shape
    return newSet

def Timer(itr=1,i=[0],t=[0]):
    import time
    itr=itr-1    
    if i[0]==1:
        timed=time.time()-t[0]                                              # calculate an estimate of total run time before done
        mins=int((itr*timed)/60)
        secs=int((itr*timed)%60)
        print("Estimate until done: {} minutes {} seconds".format(mins,secs))
        i[0]=0
        t[0]=0
    else: 
        t[0]+=time.time()
        i[0]+=1 # mutable variable get evaluated ONCE
    return 
