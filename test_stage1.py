import numpy as np
import pandas as pd
from glob import glob
import os
import scipy.ndimage
import resnet1
import cluster
import image
import logging
import time
import csv
import copy
from skimage import morphology
from scipy import ndimage
from scipy.ndimage.measurements import center_of_mass

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%Y %H:%M:%S',
                filename=time.strftime("log/%Y-%m-%d-%H:%M:%S", time.localtime())+'.log',
                filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

mode='test2'

file_list = glob('./%s/*/*.mhd' % mode)
missed = 0
miss_list = []
noded = 0 

save_train=1
image_extension=0
border=np.array([28,28,15])
scale=np.array([2,2,1],dtype=int)

if save_train:
    net2train_path='Cache/test2_stage1_merge_masklarge/'
    if not os.path.exists(net2train_path):
        os.mkdir(net2train_path)	


append = 0 ## append mode or write mode

 
index_file = -1        
for f in file_list:
    index_file+=1
    print "     "
    name = f.split('/')[-1].split('.')[0]   
    if append and save_train and os.path.exists('%s/%s.npy'%(net2train_path,name)):
        continue
    else:
        if 0:
            data = image.read_mhd(f,None,mode,norm_flag=True,mask_flag=False,save=True)
            mask = np.ones_like(data[0])
        else:
            data,mask = image.read_mhd(f,None,mode,norm_flag=True,mask_flag=True,save=True)
        name = data[1]                   
        img = data[0]
        mask = mask[border[0]:-border[0],border[1]:-border[1],border[2]:-border[2]].astype(np.uint8)
        mask = scipy.ndimage.interpolation.zoom(mask, 1/scale.astype(float))
        logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "    idx: %d; name: %s" % (index_file,name))
        
        prob = resnet1.net1Predict(img,name,save=True,stridez=64,crop=border[2]*2) ## s=128 >> slower  
        logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "    img.shape: %s ;prob.shape: %s" % (str(img.shape),str(prob.shape)))
        #prob = image.nms(prob)
        #selem = morphology.disk(1)
        #for i in range(prob.shape[0]):
        #    prob[i] = morphology.erosion(prob[i],selem=selem)
        if 1:
            selem = morphology.disk(10)
            for i in range(mask.shape[2]):
                mask[:,:,i] = morphology.dilation(mask[:,:,i],selem=selem)
                        
        h=prob.shape[2]                
        prob = prob * mask[:,:,:h]
        thresh = min(max(image.get_threshold(prob,0.001),0.93),0.98)
                       
        im_threshed=copy.deepcopy(prob)
        im_threshed[im_threshed>=thresh]=1
        im_threshed[im_threshed<thresh]=0        
        label_im, nb_labels = ndimage.label(im_threshed)
        center = []
        for i in xrange(1,nb_labels+1):
            blob_i = np.where(label_im==i,1,0)
            mass = center_of_mass(blob_i)
            #center.append([mass[0]*4+32,mass[1]*4+32,mass[2]*4+32])
            candidate=mass*scale+border
            score = np.sum(blob_i*prob)/np.count_nonzero(blob_i)
            center.append([score,candidate])
        logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"    centers num:%d;thresh:%.3f" % (nb_labels,thresh))
        


        if save_train:
            net2train = []
            index = 0
            for score,c in center:
                net2train.append([c,name,score,float(index)/nb_labels])
                index += 1
            np.save('%s/%s.npy'%(net2train_path,name),net2train)
            

    

    