import pandas as pd
import numpy as np 
import seg_test as test
import scipy.ndimage
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import os

saveNet1Path = './net1Predict/'
if not os.path.exists(saveNet1Path):
	    os.mkdir(saveNet1Path)


def net1Predict(img,name,save=False,stridez=64,crop=64):
    
    #print img.shape

    res = []
    xdim=0
    ydim=0
    zdim=0
    itez = int(img.shape[2]/stridez+1) if int(img.shape[2]%stridez)>crop else int(img.shape[2]/stridez)
    for i in range(itez):
        if i< itez-1:
            data = img[:,:,i*stridez:i*stridez+stridez+crop if i*stridez+stridez+crop<img.shape[2] else -1]
        else:
            data = img[:,:,i*stridez:i*stridez+stridez+crop if i*stridez+stridez+crop<img.shape[2] else -1]
        data = (np.asarray(data,dtype='float')-128)/2
        
        X = np.reshape(data,(1,data.shape[0],data.shape[1],data.shape[2],1))
        
        y = test.predict(X)
        #print y.shape
        y=y[0,:,:,:]
        xdim = y.shape[0]
        ydim = y.shape[1]
        zdim+= y.shape[2]
        res.append(y)
    
    prob = np.zeros((xdim,ydim,zdim)) 
    zdim = 0
    for i in range(0,len(res)):
        #print res[i].shape
        prob[:,:,zdim:zdim+res[i].shape[2]] = res[i]
        zdim+=res[i].shape[2]
    
    #print prob.shape,np.max(prob)
    
    if save == True:
	if not os.path.exists(saveNet1Path):
            os.mkdir(saveNet1Path)
        np.save('%s%s.npy'%(saveNet1Path,name),prob)
    
    return prob
