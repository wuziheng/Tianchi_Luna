import numpy as np
from glob import glob
import image
import scipy.ndimage

def resize(img,target_shape):
    real_resize = target_shape / np.array(img.shape)
    img = scipy.ndimage.interpolation.zoom(img, real_resize)
    return img

TARGET_SPACING = [0.70216,0.70216,1.1163]
Z_RATE=TARGET_SPACING[2]/TARGET_SPACING[1]

mode = 'train'
print mode
file_list = glob('/home/lijunying/luna17/Cache/net1_out_%s/LKDS-*.npy' % mode)
data = []
a=1
shape_2d=1
true_data = []
part_remove = 1
border=np.array([28,28,15])
scale=np.array([2,2,1],dtype=int)
aug_3 = True
for f in file_list:
    print f
    fname = f.split('/')[-1]
    img = np.load('/home/lijunying/luna17/Cache/normed_%s/%s'%(mode,fname))[0]
    prob = np.load('/home/lijunying/luna17/Cache/prob_%s/%s'%(mode,fname))
    coords = np.load(f) 
    #for c,name,nodes in coords:

    if coords.shape[0]==0:
        continue
    nodes=coords[0][2]
    tp_list=np.zeros([nodes.shape[0]])
    for i in range(coords.shape[0]):
        c,name,nodes,score,rank = coords[i]
        ## part remove
        if part_remove ==1:
            if rank > 0.8 and coords.shape[0]>16:
                continue
            elif rank > 0.9 and coords.shape[0]>10:
                continue
        
        label=False
        shift = 8
        remove_flag=0 ## remove more center of a node
        for j in range(nodes.shape[0]):
            node = nodes[j]
            diameter = node[1]
            node_center = node[0]
            dif = node_center - c
            dif[2]*=Z_RATE
            r=np.sqrt(np.sum(dif**2))
            if diameter[0] > r:
                label=True
                shift = min(int((diameter[0] - r) / Z_RATE),8)
                if tp_list[j]==1:
                    remove_flag=1
                else:
                    tp_list[j]=1
                break
            elif diameter[0] + 8 * Z_RATE > r:
                shift = min(int((r - diameter[0]) / Z_RATE),8)  
                
        if remove_flag==0:
            c = np.round(c).astype(int)
            if shape_2d==1:
                t  = image.crop(img,c,32+shift)
                
                c_prob = (c - border) / scale
                prob_cut  = image.crop(prob,c_prob,(32+shift)/2)[1]
                
                if t[0] == True:
                    _slice =t[1][:,:,32+shift].astype(np.uint8)   
                    #t_prob = resize(prob_cut[:,:,16+shift/2],np.array([64+2*shift,64+2*shift],dtype=float))
                    #out_img = np.transpose(np.array([_slice,t_prob]),[2,1,0])
                    data.append([_slice,label,name,nodes])
                    if label == True and mode =='train' and aug_3==True:
                        _slice =t[1][:,:,31+shift].astype(np.uint8)   
                        #t_prob = resize(prob_cut[:,:,15+shift/2],np.array([64+2*shift,64+2*shift],dtype=float))
                        #out_img = np.transpose(np.array([_slice,t_prob]),[2,1,0])
                        data.append([_slice,label,name,nodes])
                        
                        _slice =t[1][:,:,33+shift].astype(np.uint8)   
                        #t_prob = resize(prob_cut[:,:,17+shift/2],np.array([64+2*shift,64+2*shift],dtype=float))
                        #out_img = np.transpose(np.array([_slice,t_prob]),[2,1,0])
                        data.append([_slice,label,name,nodes])
                    
                    if 0:
                        _slice =t[1][:,:,32+shift] 
                        data.append([_slice.astype(np.uint8),label,name,nodes])
                        _slice =t[1][:,:,31+shift] 
                        data.append([_slice.astype(np.uint8),label,name,nodes])
                        _slice =t[1][:,:,33+shift] 
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
            
    if 0:
        for node in nodes:
            node_center = node[0]
            diameter = node[1]
            shift = min(int((diameter[0]) / Z_RATE),8)
            c = np.round(node_center).astype(int)
            t  = image.crop(img,c,29+shift)
            label = True
            
            if t[0] == True:
                if shape_2d==1:
                    _slice =t[1][:,:,32+shift:33+shift]   
                else:
                    _slice =t[1][:,:,13+shift/2:45+shift/2]
                #data.append([_slice.astype(np.uint8),label,name])
            #true_data.append([_slice.astype(np.uint8),label,name])
        

print len(data)        
np.save('data_classify_%s_large_mask.npy' % mode,data)
#np.save('true_data.npy',true_data)
