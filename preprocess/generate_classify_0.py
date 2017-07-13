import numpy as np
from glob import glob
import image
TARGET_SPACING = [0.70216,0.70216,1.1163]
Z_RATE=TARGET_SPACING[2]/TARGET_SPACING[1]

mode = 'train'

file_list = glob('/home/lijunying/luna17/Cache/net1_out_%s/LKDS-*.npy' % mode)
data = []
a=1
true_data = []
for f in file_list:
    print f
    fname = f.split('/')[-1]
    img = np.load('/home/lijunying/luna17/Cache/normed_%s/%s'%(mode,fname))[0]
    coords = np.load(f) 
    #for c,name,nodes in coords:
        

    if coords.shape[0]==0:
        continue
    
    for i in range(coords.shape[0]):
        c,name,nodes = coords[i]
        label=False
        shift = 8
        for node in nodes:
            diameter = node[1]
            node_center = node[0]
            dif = node_center - c
            dif[2]*=Z_RATE
            r=np.sqrt(np.sum(dif**2))
            if diameter[0] > r:
                label=True
                shift = min(int((diameter[0] - r) / Z_RATE),8)
                break
            elif diameter[0] + 8 * Z_RATE > r:
                shift = min(int((r - diameter[0]) / Z_RATE),8)                
        c = np.round(c).astype(int)
        if 0:
            t  = image.crop(img,c,32+shift)
            if t[0] == True:
                _slice =t[1][:,:,31+shift:34+shift]   
    
                data.append([_slice.astype(np.uint8),label,name,nodes])
                if label ==True:
                    true_data.append([_slice.astype(np.uint8),label,name,nodes])
        else:
            t  = image.crop(img,c,29+shift)
            if t[0] == True:
                _slice =t[1][:,:,13+shift/2:45+shift/2]       
                data.append([_slice.astype(np.uint8),label,name,nodes])
                if label ==True:
                    true_data.append([_slice.astype(np.uint8),label,name,nodes])
            

    for node in nodes:
        node_center = node[0]
        diameter = node[1]
        shift = min(int((diameter[0]) / Z_RATE),8)
        c = np.round(node_center).astype(int)
        t  = image.crop(img,c,29+shift)
        label = True
        if t[0] == True:
            _slice =t[1][:,:,13+shift/2:45+shift/2]   
            data.append([_slice.astype(np.uint8),label,name])
        #true_data.append([_slice.astype(np.uint8),label,name])
        

print len(data)        
np.save('data_classify_%s_v1.npy' % mode,data)
#np.save('true_data.npy',true_data)
