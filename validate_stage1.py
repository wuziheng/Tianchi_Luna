import numpy as np
import pandas as pd
from glob import glob
import os
import scipy.ndimage
import resnet1
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

mode='train'

file_list = glob('./%s/*/*.mhd' % mode)
cads = pd.read_csv('./csv/%s/annotations.csv' % mode)

save_train=1
image_extension=0
save_path='Cache/net1Predict/'
wrong_path=save_path+'wrong_npy/'

model='resnet'
if model=='resnet':
    border=np.array([28,28,15])
    scale=np.array([2,2,1],dtype=int)
elif model=='resnet_seg_fat':
    border=np.array([28,28,13])
    scale=np.array([2,2,1],dtype=int)



if save_train:
    net2train_path='Cache/net1_out_%s' % mode
    prob_path = 'Cache/prob_%s' % mode
    if not os.path.exists(net2train_path):
        os.mkdir(net2train_path)	
    if not os.path.exists(prob_path):
        os.mkdir(prob_path)	
if not os.path.exists(save_path):
        os.mkdir(save_path)	
if not os.path.exists(wrong_path):
        os.mkdir(wrong_path)	

append = 0 ## append mode or write mode
top_num_list = np.zeros([10])

with open(save_path+'wrong_%s.csv' % mode, 'wb') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['id','y','x','z','prob','thresh','r','n_y','n_x','n_z','distance'])    
    total_num=0
    total_find=0
    total_cluster=0  
    total_sample=0  
    index_file = -1        
    for f in file_list:
        index_file+=1      

        name = f.split('/')[-1].split('.')[0]
        logging.info("-------------------------------------------")
        logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "    idx: %d; name: %s; pid: %s" % (index_file,name,os.getpid()))
        if append and save_train and os.path.exists('%s/%s.npy'%(net2train_path,name)):
            continue
        else:
            if 0:
                data = image.read_mhd(f,cads,mode,norm_flag=True,mask_flag=False,save=True)
                mask = np.ones_like(data[0])
            else:
                data,mask = image.read_mhd(f,cads,mode,norm_flag=True,mask_flag=True,save=True)
            name = data[1]
            nodes = data[2]

            mask = mask[border[0]:-border[0],border[1]:-border[1],border[2]:-border[2]].astype(np.uint8)
            mask = scipy.ndimage.interpolation.zoom(mask, 1/scale.astype(float))
            
            img = data[0]
                            
            if model=='resnet':
                prob = resnet1.net1Predict(img,name,save=True,stridez=64,crop=border[2]*2) ## s=128 >> slower 
            else:
                try:
                    if img.shape[0]>600:
                        raise Exception('Memory is limited, input image is too large(%s), split it into two.' %(str(img.shape)))
                    else:
                        prob = resnet1.net1Predict(img,name,save=True,stridez=32,crop=border[2]*2) ## s=128 >> slower 
                except Exception as err:
                    logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "    %s" % err)
                    w=img.shape[0]/2
                    prob1 = resnet1.net1Predict(img[:w+border[0]],name,save=True,stridez=32,crop=border[2]*2)
                    prob2 = resnet1.net1Predict(img[w-border[0]:],name,save=True,stridez=32,crop=border[2]*2)
                    prob = np.concatenate((prob1,prob2),axis=0)
                
                
            if save_train:
                np.save('%s/%s.npy'%(prob_path,name),(prob*255).astype(np.uint8))
            logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "    img.shape: %s ;prob.shape: %s;mask.shape: %s" % (str(img.shape),str(prob.shape),str(mask.shape)))
            #prob = image.nms(prob)
            if 1:
                selem = morphology.disk(2)
                for i in range(mask.shape[2]):
                    mask[:,:,i] = morphology.dilation(mask[:,:,i],selem=selem)
            h=prob.shape[2]                
            prob = prob * mask[:,:,:h]
            #thresh = min(max(image.get_threshold(prob,0.001),0.93),0.98)
            thresh = 0.93
            ## print real nodes
            for node in nodes:
                r = node[1][0]
                node_ori=np.array(node[0],dtype=int)

                node = (node_ori-border)/scale
                try:
                    logging.info("node_origingal:%s,node:%s,prob:%.4f,mask:%d,r:%.1f"%(str(node_ori),str(node),prob[node[0]][node[1]][node[2]],mask[node[0]][node[1]][node[2]],r))
                    if mask[node[0]][node[1]][node[2]]!=1:
                        logging.info("Masked!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
                except:
                    logging.info("node:"+str(node)+'prob: out of bound')               
            # threshold and cluster
            im_threshed=copy.deepcopy(prob)
            im_threshed[im_threshed>=thresh]=1
            im_threshed[im_threshed<thresh]=0
            if 0:
                selem = morphology.disk(1)
                for i in range(prob.shape[0]):
                    im_threshed[i] = morphology.erosion(im_threshed[i],selem=selem)
             
            label_im, nb_labels = ndimage.label(im_threshed)
            center = []
            candidate_list = []
            
            for i in xrange(1,nb_labels+1):
                blob_i = np.where(label_im==i,1,0)
                mass = center_of_mass(blob_i)
                candidate=mass*scale+border
                score = np.sum(blob_i*prob)/np.count_nonzero(blob_i)
                center.append([score,candidate])
                candidate_list.append(candidate)
            logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"    centers num:%d;thresh:%.3f" % (nb_labels,thresh))
            #center.sort(lambda a,b: int(b[1]>a[1]))
            try:
                center.sort(reverse=True)
            except:
                logging.info("Sort Failed!!!!!!!!!!!!!!")    
                    #center.reverse()
                    

            net2train = []
            find_array = np.zeros([len(nodes)])
            index= 0
            for score,c in center:                    
                label = False
                for i in range(len(nodes)):
                    diameter = nodes[i][1]
                    node = nodes[i][0]
                    dif=np.abs(c-node)
                    if  dif[0]<diameter[0]/2 and dif[1]<diameter[1]/2 and dif[2]<diameter[2]/2:
                    #if  dif[0]<40 and dif[1]<40 and dif[2]<40:
                        label = True
                        logging.info('Found: center:[%.1f %.1f %.1f];node:[%.1f %.1f %.1f];score:%.4f;rank:%d' %( c[0], c[1], c[2], node[0], node[1], node[2],score,index+1))
                        if find_array[i]==0:
                            top_num_list[index*10/nb_labels:] +=1
                        find_array[i]=1
                net2train.append([c,name,nodes,score,float(index)/nb_labels])
                index +=1
                
            if save_train:
                np.save('%s/%s.npy'%(net2train_path,name),net2train)
            find= np.count_nonzero(find_array)
            cant=0
            miss =len(nodes)-find
        
                    
            total_num+=len(nodes)
            total_find+=find
            total_cluster+=nb_labels
            total_sample+=1
            logging.info("ID:%s,find:%d,cant:%d,miss:%d"%(name,find,cant,miss))
            logging.info("luna num:%d,total_find:%d,recall:%f,average_cluster:%d"%(total_num,total_find,float(total_find)/total_num,total_cluster/total_sample))
            tmp ='['
            for t in top_num_list:
                tmp +='%.4f ' % (float(t)/total_num)
            logging.info("TOP_list: %s]"%tmp)
            
            
            
            if miss>0:
                #np.save('%s%s.npy'%(wrong_path,name),prob)
                candidate_list=np.array(candidate_list)
                for node in nodes:
                    tmp=np.abs(candidate_list-node[0])
                    tmp=np.sqrt(np.sum(tmp**2,1))
                    index=np.where(tmp==np.min(tmp))
                    node_ori=node[0]
                    r = node[1][0]
                    node_prob = ((node_ori-border)/scale).astype(int)
                    logging.info('nearst candidate: distance:%.4f' % np.min(tmp) + ' candidate:' + str(candidate_list[index][0]) + ' node:' + str(node[0]))
                    try:
                        spamwriter.writerow([name,node_ori[0],node_ori[1],node_ori[2],prob[node_prob[0]][node_prob[1]][node_prob[2]],thresh,r,candidate_list[index][0][0],candidate_list[index][0][1],candidate_list[index][0][2],np.min(tmp)])
                    except:
                        spamwriter.writerow([name,node_ori[0],node_ori[1],node_ori[2],'out of bound',thresh,r,candidate_list[index][0][0],candidate_list[index][0][1],candidate_list[index][0][2],np.min(tmp)])

logging.info("All work done!")

   
