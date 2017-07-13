import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import time
import glob
import pandas as pd
import image
TARGET_SPACING = [0.70216,0.70216,1.1163]
Z_RATE=TARGET_SPACING[2]/TARGET_SPACING[1]
def get_label(img,imageName,nodes,get_label=False):
    #time0=time.time()
    if get_label:
        label=np.zeros_like(img,dtype=float)
    dots=[]
    for node in nodes:
        coord=node[0]
        diameter=node[1]       
        dot_base=np.floor(coord-diameter/2).astype(int)
        dot_num=0
        for i in range(int(diameter[0]+1)):
            for j in range(int(diameter[1]+1)):
                for k in range(int(diameter[2]+1)):
                    dot=dot_base+np.array([i,j,k])
                    tmp=dot-coord
                    tmp[2]*=Z_RATE
                    r=np.sqrt(np.sum(tmp**2)) 
                    if r < diameter[0]/2:
                        weight=1-2*float(r)/diameter[0]
                        dot_num+=1
                        dots.append([dot,weight])
                        #print 
                        if get_label:
                            label[dot[0],dot[1],dot[2]]=1-2*r/diameter[0]
        for i in range(-dot_num,0):
            dots[i][1]/=dot_num
        
    if get_label:
        return label
    else:
        return dots
    
    #plt.imshow(label[20:40,220:240,93])
		
if __name__ == "__main__":
    if 0:
        img,imageName,nodes=np.load('LKDS-00005.npy')
        label=get_label(img,imageName,nodes)
        plt.imshow(label[20:40,220:240,93])
        #plt.imshow(img[20:40,220:240,93])
    else:
        mode = 'val'
        
        file_list = glob.glob('../%s/*/*.mhd' % mode)
        cads = pd.read_csv('../csv/%s/annotations.csv' % mode)
        savedata=[]
        for f in file_list:           
            img,imageName,nodes = image.read_mhd(f,cads,mode)
            dots=get_label(img,imageName,nodes,get_label=False)
            savedata.append([img,dots,imageName,nodes])
        np.save('data_seg_%s.npy' % mode,savedata)
            
            