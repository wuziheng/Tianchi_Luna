import argparse
import tensorflow as tf
import numpy as np
import time
import csv
import os
import random
from sklearn.metrics import precision_score,recall_score
import logging
from random import randint
import copy
from scipy import ndimage
from scipy.ndimage.measurements import center_of_mass
from skimage import morphology
## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='resnet', type=str)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--display', default=100, type=int)
parser.add_argument('--save_every', default=5, type=int)
parser.add_argument('--epoch', default=50, type=int)
parser.add_argument('--decay_epoch', default=18, type=int)
parser.add_argument('--begin_step', default=13300, type=int)
parser.add_argument('--learning_rate', default=1e-3, type=float)
parser.add_argument('--boosting_ratio', default=1.2, type=float)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--validate_size', default=100, type=int)
parser.add_argument('--num_negtive', default=10, type=int)
parser.add_argument('--shuffle', default=True, type=bool) 
parser.add_argument('--data_path', default='data/data_seg.npy', type=str)
parser.add_argument('--boosting_mode', default='MergeBoosting_new', type=str)
parser.add_argument('--neg_max_weight', default=0, type=float)
parser.add_argument('--keep_rate', default=0.8, type=float)
args = parser.parse_args()

if args.model=='seg':
	import models.seg as model
elif args.model=='resnet':
	import models.resnet as model
elif args.model=='resnet_seg_fat':
	import models.resnet_seg_fat as model
else:
	print 'no corresponding model'
	import models.seg as model

if args.model=='resnet' or args.model=='linet':
    border=np.array([28,28,15])
    scale=np.array([2,2,1],dtype=int)
    crop_size=np.array([120,120,70])
elif args.model=='seg_2d':
    border=np.array([28,28,0])
    scale=np.array([2,2,1],dtype=int)
    crop_size=np.array([120,120,40])
elif args.model=='unet':
    border=np.array([16,16,12])
    scale=np.array([1,1,1],dtype=int)
    crop_size=np.array([80,80,80])
elif args.model=='resnet_seg_fat':
    border=np.array([28,28,13])
    scale=np.array([2,2,1],dtype=int)
    crop_size=np.array([120,120,66])  
elif args.model=='seg_nopolling':
    border=np.array([27,27,14])
    scale=np.array([1,1,1],dtype=int)
    crop_size=np.array([120,120,70])  

    
   
savePath = './snapshot/save_'+ args.model + '_' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + '/'
if not os.path.exists(savePath):
        os.mkdir(savePath)	
## log	
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%Y %H:%M:%S',
                filename=savePath+'info.log',
                filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-8s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
## args
logging.info(args)
batch_size = args.batch_size # batch_size <=4 , for memory constrain
display = args.display
hm_epochs = args.epoch
save_every = args.save_every
boosting_ratio = args.boosting_ratio
boosting_mode =args.boosting_mode
num_negtive=args.num_negtive

x = tf.placeholder('float',shape=[batch_size,crop_size[0],crop_size[1],crop_size[2],1])
y = tf.placeholder('float')
w = tf.placeholder('float') 

def get_data_list(data,num_negtive,crop_size,border,scale,try_max=50):
    # input img:     crop_size
    # output_img:    core = (cropsize - 2*border) / scale
    # input_effect:  effect_size = cropsize - 2 * border
    # min_distance:  min_distance =effect_size / 2
    # coord_out =    (coord_in - border) / scale
    
    data_list=[]
    min_distance=crop_size/scale-border
    core=crop_size-2*border
    for i in range(data.shape[0]):
        #print i 
        img=data[i,0]
        nodes=data[i,3]
        dots=data[i,1]
        ## append positive
        weight_sum=0
        for dot, weight in dots:
            weight_sum+=weight
        pos_coords=[]
        for node in nodes:
            coord_begin = node[0].astype(int) -crop_size/2
            nodes_in = []               
            for j in range(len(nodes)):
                dif = nodes[j][0] - node[0] # Do not change it into integer. keey accuracy
                if np.max(np.abs(dif) - min_distance) < 0:
                    coord_label = (nodes[j][0] - coord_begin - border) / scale ## luna coord of label space
                    diameter = nodes[j][1] / scale                         ## luna diameter of label space
                    nodes_in.append([coord_label,diameter])                   
            #positive_rate=float(len(dots))/len(nodes)/core[0]/core[1]/core[2]
            positive_rate=float(weight_sum)/len(nodes)/core[0]/core[1]/core[2]           
            data_list.append([i, coord_begin, positive_rate, nodes_in])
            pos_coords.append(coord_begin)
        ## append negtive
        j=0
        try_nums=0
        while j < num_negtive:
            try_nums+=1
            if try_nums>try_max:
                break
            random_size=np.array(img.shape)-np.array(crop_size)
            coord_begin=np.array([randint(0,random_size[0]-1),randint(0,random_size[1]-1),randint(0,random_size[2]-1)])
            ok_flag=1
            for pos_coord in pos_coords:
                dif=np.abs(coord_begin-pos_coord)
                if dif[0]<min_distance[0] or dif[1]<min_distance[1] or dif[2]<min_distance[2]:
                    ok_flag=0
                    break
            if ok_flag:
                data_list.append([i, coord_begin, 0, []])
                j+=1
            
    return data_list


## load data    
data_path=args.data_path
logging.info( '['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'  loading data from '+ data_path)
use_val = True
if not use_val:
    data = np.load(data_path)
    train_data = data[args.validate_size:]
    validation_data = data[:args.validate_size]
else:
    train_data = np.load(data_path)   
    logging.info( '['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'  loading data from '+ 'data/data_seg_val.npy')
    validation_data = np.load('data/data_seg_val.npy')
logging.info( '['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'  loading completed!')


# This shuffle process is canceled for keeping validation data identical
if args.shuffle == True and use_val:
	index=range(train_data.shape[0])
	random.shuffle(index)
	train_data=train_data[index]
	index=range(validation_data.shape[0])
	random.shuffle(index)
	validation_data=validation_data[index]


logging.info( '['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '  border: %s, scale: %s, crop_size: %s' % (str(border),str(scale),str(crop_size)))
core=(crop_size-2*border)/scale
#train_len=(train_data.shape[0]/batch_size)*batch_size  # make it times of batch_size
#train_data=train_data[0:train_len]    

train_data_list=get_data_list(train_data,num_negtive,crop_size,border,scale)
train_len=(len(train_data_list)/batch_size)*batch_size  # make it times of batch_size
validation_data_list=get_data_list(validation_data,num_negtive,crop_size,border,scale)

if args.shuffle == True:
	index=range(len(train_data_list))
	random.shuffle(index)
	old=copy.deepcopy(train_data_list)
	train_data_list=[]
	for i in index:
            train_data_list.append(old[i])
	logging.info( '['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'  shuffuling completed')
    
np.save('data/train_data_list.npy',np.array(train_data_list))
np.save('data/val_data_list.npy',np.array(validation_data_list))

i,coord_begin,pos_rate,nodes_in=zip(*train_data_list)
train_pos_num=np.sum(float(pos_rate>0))
train_positive_rate=sum(pos_rate)/len(pos_rate)
i,coord_begin,pos_rate,nodes_in=zip(*validation_data_list)
val_positive_rate=sum(pos_rate)/len(pos_rate)
val_pos_num=np.sum(float(pos_rate>0))
train_scans_num=len(set(train_data[:,2]))
validation_scans_num=len(set(validation_data[:,2]))
neg_rate_list=[0.125,0.25,0.5,1,2,4,8]
logging.info( '['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'  train_data.shape:'+str(len(train_data_list))+' positive rate:' + str(train_positive_rate))
logging.info( '['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'  validation_data.shape:'+str(len(validation_data_list))+' positive rate:' + str(val_positive_rate))

	


            
def get_label(dots,begin_dot,crop_size,border):
    begin_label=begin_dot+border
    end_label=begin_dot+crop_size-border
    w=np.zeros(core,dtype=float)
    for dot,weight in dots:
        if np.min(dot-begin_label) > 0 and np.min(end_label-dot)>0:
            new_coord=((dot-begin_label)/scale).astype(int)
            w[new_coord[0],new_coord[1],new_coord[2]]=1
    y=w>0
    w=w*((1-train_positive_rate)/train_positive_rate-1)+1
    #w=w+train_positive_rate
    return w,y

def get_label_from_nodes(nodes,begin_dot,crop_size,border,neg_max_weight=0.0):
    begin_label=begin_dot+border
    end_label=begin_dot+crop_size-border
    w=np.zeros(core,dtype=float)
    y=np.zeros(core,dtype=float)
    TARGET_SPACING = [0.70216,0.70216,1.1163]
    Z_RATE=TARGET_SPACING[2]/TARGET_SPACING[1]
    
    for node in nodes:
        coord=node[0]
        diameter=node[1]       
        begin_node=np.floor(coord-diameter).astype(int)
        end_node=np.floor(coord+diameter).astype(int)
        
        if np.min(begin_node-end_label) > 0 or np.min(begin_label-end_node)>0:
            continue
        
        begin_node = np.maximum(begin_node - begin_label, np.array([0,0,0]))
        end_node = np.minimum(end_node - begin_label, core*scale)
        
        ## large node weight decay
        ## goal: prevent from little node misclassified

        decay = min( 10**3 / diameter[0]**3, 1)
        for i in range(begin_node[0], end_node[0]):
            for j in range(begin_node[1], end_node[1]):
                for k in range(begin_node[2], end_node[2]):
                    dot=begin_label+np.array([i,j,k])
                    tmp=dot-coord
                    tmp[2]*=Z_RATE
                    r=np.sqrt(np.sum(tmp**2))                   
                    if r <= diameter[0]/2:
                        ## distance weight decay
                        ## goal: detect positive nodes more precisely
                        ## it works, but we don't compare it with setting the same weight 
                        w[i/scale[0], j/scale[1], k/scale[2]] = max((1-(2 * float(r) / diameter[0])) * decay, w[i/scale[0], j/scale[1], k/scale[2]])
                        y[i/scale[0], j/scale[1], k/scale[2]] = 1
                    ## node outside weight boosting
                    ## goal: prevent from locating nodes wrongly
                    ## But we find it seems useless
                    elif r > diameter[0] and r < 2 *diameter[0]:
                        if y[i/scale[0], j/scale[1], k/scale[2]]==0:
                            w[i/scale[0], j/scale[1], k/scale[2]] = (((2*float(r) - diameter[0]) / diameter[0])) * neg_max_weight
                                  
    #print 'node:',nodes,';w_sum:',np.sum(w)
    w=w*((1-train_positive_rate)/train_positive_rate-1)+(1-8*neg_max_weight)
    return w,y

def get_normalized_data_list(iters,batch_size,data,data_list,crop_size,border,weight,epoch,neg_finetune=0,weight_scale=None,train_y=None):    
    X=[]
    Y=[]
    W=[]
    core=(crop_size-2*border)/scale
    for i in range(iters*batch_size,(iters+1)*batch_size):

        index, begin_coord, positive_rate, node_in = data_list[i]
        img=data[index,0]
        #print 'img:',img.shape
        #dots=data[index,1]
        nodes=data[index,3]
        if positive_rate==0:
            end_coord=begin_coord+crop_size
            X.append(img[begin_coord[0]:end_coord[0],begin_coord[1]:end_coord[1],begin_coord[2]:end_coord[2]])
            y=np.zeros([core[0],core[1],core[2]])
            if epoch==0:
                w=y            
            #print 'a:',begin_coord,end_coord
        else:
            end_coord=begin_coord+crop_size
            if np.min(begin_coord)>=0 and end_coord[0]<=img.shape[0] and end_coord[1]<=img.shape[1] and end_coord[2]<=img.shape[2]:
                X.append(img[begin_coord[0]:end_coord[0],begin_coord[1]:end_coord[1],begin_coord[2]:end_coord[2]])
                #print 'b:',begin_coord,end_coord
            else:
                x=np.zeros([crop_size[0],crop_size[1],crop_size[2]])
                real_begin_coord = np.maximum(begin_coord,0)
                real_end_coord = np.minimum(end_coord,img.shape)
                x_begin_coord = real_begin_coord - begin_coord
                x_end_coord = real_end_coord - real_begin_coord + x_begin_coord
                #print 'c:',begin_coord,end_coord,real_begin_coord,real_end_coord,x_begin_coord,x_end_coord
                x[x_begin_coord[0]:x_end_coord[0],x_begin_coord[1]:x_end_coord[1],x_begin_coord[2]:x_end_coord[2]] = img[real_begin_coord[0]:real_end_coord[0],real_begin_coord[1]:real_end_coord[1],real_begin_coord[2]:real_end_coord[2]]
                
                X.append(x)
            #w,y = get_label(dots,begin_coord,crop_size,border)
            if epoch==0:
                w,y = get_label_from_nodes(nodes,begin_coord,crop_size,border,neg_max_weight=args.neg_max_weight)
             
        if epoch==0:
            Y.append(y)
            W.append(w) 
        
    X=np.reshape(X,[-1,crop_size[0],crop_size[1],crop_size[2],1]) 
    X=(X.astype(float)-128)/2 
    #X=(X.astype(float)/128) -1
    if epoch==0:
        weight[iters*batch_size:(iters+1)*batch_size]=np.array(W)
        if train_y is not None:
            train_y[iters*batch_size:(iters+1)*batch_size]=np.array(Y)
    else:
        W=weight[iters*batch_size:(iters+1)*batch_size]
        if train_y is None:
            Y=W > (1 - 8 * args.neg_max_weight)
        else:
            Y=train_y[iters*batch_size:(iters+1)*batch_size]
        Y=np.reshape(Y,[-1,core[0],core[1],core[2]]) 
        
        if neg_finetune!=0 and neg_finetune!=1:
            W=W+W*(1-Y)*(neg_finetune-1)
            weight[iters*batch_size:(iters+1)*batch_size]=np.array(W)
            
    if weight_scale is not None:
        for i in range(batch_size):
            W[i] *= weight_scale[iters*batch_size + i]
        #W=W*np.reshape(weight_scale[iters*batch_size:(iters+1)*batch_size],[batch_size])
    
    return X,Y,W

def pos_detect(batch_size,data_list,conf,Y,thresh=0.5,weight_scale=None,boosting_ratio=1.0):
    TP,NP,P=0,0,0
    y_shape=np.array(Y).shape
    num = y_shape[0]*y_shape[1]*y_shape[2]*y_shape[3]
    for j in range(1,25):
        if np.count_nonzero(conf>1-j*0.02) > 0.01 * num:
            thresh = 1-j*0.02
            break
    
    for i in range(batch_size):
        im=conf[i]
        y=Y[i]
        im[im>=thresh]=1
        im[im<thresh]=0 
        selem = morphology.disk(1)
        for j in range(im.shape[0]):
            im[j] = morphology.erosion(im[j],selem=selem)
          
          
        label_im, nb_labels = ndimage.label(im)
        nodes = data_list[i][3]
        p_sample=len(nodes)
        tp_sample=np.zeros([p_sample])
        np_sample=0
        #print i,nodes
        for j in xrange(1,nb_labels+1):
            blob_i = np.where(label_im==j,1,0)
            mass = center_of_mass(blob_i)
            #print mass
            if y[int(round(mass[0])),int(round(mass[1])),int(round(mass[2]))]>0 and p_sample > 0:
                for k in range(p_sample):
                    node_coord = nodes[k][0]
                    diameter = nodes[k][1]
                    dif = mass - node_coord
                    ## This is a simple judge. Not perfect
                    if np.max(np.abs(dif)-diameter/2) <0:
                        tp_sample[k]=1                  
            else:               
                np_sample+=1
            
        #print tp_sample,np_sample,p_sample
        P+=p_sample
        TP+=np.count_nonzero(tp_sample)
        NP+=np_sample
        if weight_scale!=None:
            #weight_scale[i] *= 1/boosting_ratio if np.count_nonzero(tp_sample) == p_sample and np_sample==0 else boosting_ratio
            weight_scale[i] *= 1/boosting_ratio if np.count_nonzero(tp_sample) == p_sample and np_sample==0 else boosting_ratio
            
    if weight_scale!=None:
        return TP,NP,P,weight_scale
    else:
        return TP,NP,P      			
def Adaboosting(conf,Y,W,boosting_ratio=1.15,thresh=0.5):
    if boosting_ratio==1:
        return W
    
    judge =  ((conf > thresh) == Y).astype(float)
    W=W/(np.maximum(judge*boosting_ratio,1))
    W=W*(np.maximum((1-judge)*boosting_ratio,1))    
    return W

def pos_detect_MergeBoosting(batch_size,data_list,conf,Y,W,boosting_ratio=1.0,thresh=0.5):
    TP,NP,P=0,0,0
    y_shape=np.array(Y).shape
    num = y_shape[0]*y_shape[1]*y_shape[2]*y_shape[3]
    for j in range(1,25):
        if np.count_nonzero(conf>1-j*0.02) > 0.01 * num:
            thresh = 1-j*0.02
            break
    
    for i in range(batch_size):
        im=conf[i]
        y=Y[i]
        w=W[i]
        im[im>=thresh]=1
        im[im<thresh]=0        
        label_im, nb_labels = ndimage.label(im)
        nodes = data_list[i][3]
        p_sample=len(nodes)
        tp_sample=np.zeros([p_sample])
        np_sample=0
        #print i,nodes
        for j in xrange(1,nb_labels+1):
            blob_i = np.where(label_im==j,1,0)
            mass = center_of_mass(blob_i)
            #print mass
            if y[int(round(mass[0])),int(round(mass[1])),int(round(mass[2]))]>0 and p_sample > 0:
                for k in range(p_sample):
                    node_coord = nodes[k][0]
                    diameter = nodes[k][1]
                    dif = mass - node_coord
                    ## This is a simple judge. Not perfect
                    if np.max(np.abs(dif) - diameter / 2) < 0:
                        tp_sample[k]=1   
                           
                # pos decay
                if boosting_ratio!=1:
                    w /= np.maximum( boosting_ratio * (blob_i==1) * (y ==1) , 1)
                    
            else:               
                np_sample+=1               
                # neg boost
                if boosting_ratio!=1:                  
                    w = np.maximum( np.maximum(w * boosting_ratio, 1/train_positive_rate) * (blob_i==1) * (y==0) , w)  
            
        
        if boosting_ratio!=1:
            # pos boost
            for k in range(len(tp_sample)):
                if(tp_sample[k]==0):
                    node_coord = nodes[k][0].astype(int)
                    diameter = nodes[k][1].astype(int)
                    for k_x in range(-diameter[0],diameter[0]):
                        for k_y in range(-diameter[1],diameter[1]):
                            for k_z in range(-diameter[2],diameter[2]):
                                dot = [k_x, k_y, k_z] + node_coord
                                if np.min(dot) >= 0 and np.max(dot-np.array(y.shape)) < 0 and y[dot[0],dot[1],dot[2]] != im[dot[0],dot[1],dot[2]]:
                                    w[dot[0], dot[1], dot[2]] *= boosting_ratio 
        
            #if np.count_nonzero(tp_sample) == p_sample and np_sample ==0:
            #    w /= (boosting_ratio-1)/3 +1
            #    w /= np.maximum((boosting_ratio * (y==0)),1)
            # neg decay
            if np_sample==0:
                #w /= (boosting_ratio-1)/3 +1
                w /= np.maximum(((boosting_ratio-1)/4 +1) * (y==0),1)
        #print tp_sample,np_sample,p_sample
        P+=p_sample
        TP+=np.count_nonzero(tp_sample)
        NP+=np_sample
        #if np.count_nonzero(tp_sample) < p_sample or np_sample > 0:
            #w = Adaboosting(im,y,w,boosting_ratio=boosting_ratio,thresh=thresh)
            
    return TP,NP,P,W 
    

def pos_detect_MergeBoosting_new(batch_size,data_list,conf,Y,W,FP_rate,FN_rate,boosting_ratio=1.0,thresh=0.5):
    TP,NP,P=0,0,0
    y_shape=np.array(Y).shape
    num = y_shape[0]*y_shape[1]*y_shape[2]*y_shape[3]
    num_pixels = y_shape[1]*y_shape[2]*y_shape[3]
    for j in range(1,10):
        if np.count_nonzero(conf>1-j*0.02) > 0.01 * num:
            thresh = 1-j*0.02
            break
    
    for i in range(batch_size):
        im=conf[i]
        y=Y[i]
        w=W[i]
        im[im>=thresh]=1
        im[im<thresh]=0        
        label_im, nb_labels = ndimage.label(im)
        nodes = data_list[i][3]
        p_sample=len(nodes)
        tp_sample=np.zeros([p_sample])
        np_sample=0
        #print i,nodes
        for j in xrange(1,nb_labels+1):
            blob_i = np.where(label_im==j,1,0)
            mass = center_of_mass(blob_i)
            #print mass
            if y[int(round(mass[0])),int(round(mass[1])),int(round(mass[2]))]>0 and p_sample > 0:
                for k in range(p_sample):
                    node_coord = nodes[k][0]
                    diameter = nodes[k][1]
                    dif = mass - node_coord
                    ## This is a simple judge. Not perfect
                    if np.max(np.abs(dif) - diameter / 2) < 0:
                        tp_sample[k]=1   
                              
            else:               
                np_sample+=1               
                # neg boost
                if boosting_ratio!=1 and FP_rate > 0:
                    wrong = (blob_i==1) * (y==0)
                    wrong_num = max(np.count_nonzero(wrong),10) # 10, preventing frome the w change two much when wrong_num is too small
                    w += wrong * boosting_ratio * (1 / 2 / FP_rate) * num_pixels / wrong_num
            
        
        if boosting_ratio!=1 and FN_rate > 0:
            # pos boost
            for k in range(len(tp_sample)):
                if(tp_sample[k]==0):
                    wrong = np.zeros_like(w)
                    node_coord = nodes[k][0].astype(int)
                    diameter = nodes[k][1].astype(int)
                    for k_x in range(-diameter[0],diameter[0]):
                        for k_y in range(-diameter[1],diameter[1]):
                            for k_z in range(-diameter[2],diameter[2]):
                                dot = [k_x, k_y, k_z] + node_coord
                                if np.min(dot) >= 0 and np.max(dot-np.array(y.shape)) < 0 and y[dot[0],dot[1],dot[2]] != im[dot[0],dot[1],dot[2]]:
                                    wrong[dot[0], dot[1], dot[2]] = 1
                    wrong_num = max(np.count_nonzero(wrong),10)
                    w += wrong * boosting_ratio * (1 / 2 / FN_rate) * num_pixels / wrong_num
            w /= 1 + boosting_ratio
        P+=p_sample
        TP+=np.count_nonzero(tp_sample)
        NP+=np_sample
        #if np.count_nonzero(tp_sample) < p_sample or np_sample > 0:
            #w = Adaboosting(im,y,w,boosting_ratio=boosting_ratio,thresh=thresh)
            
    return TP,NP,P,W 
        
def train_neural_network(x):
    
    prediction=model.tf.reduce_max(model.inference_small(x,is_training=True),4)
    confidence=tf.nn.sigmoid(prediction)
    loss = tf.reduce_mean( w*tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=prediction))
    correct = tf.equal(tf.greater(prediction, 0),tf.greater(y, 0))
    accuracy = tf.reduce_sum(tf.cast(correct, 'float'))
    
    global_step = tf.Variable(0, trainable=False)
    num_decay_steps=train_len/batch_size*args.decay_epoch
    learning_rate=tf.train.exponential_decay(args.learning_rate, global_step-args.begin_step, num_decay_steps, 0.1, staircase=True)
    logging.info( 'learning_rate_decay: num_decay_steps:%d'%num_decay_steps)
    base_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
    
    UPDATE_OPS_COLLECTION = 'resnet_update_ops'
    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    optimizer = tf.group(base_optimizer, batchnorm_updates_op)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(max_to_keep=10000)    
    
    with tf.Session(config=config) as sess:
        # Init
        if args.pretrain:
            saver.restore(sess,args.pretrain)
            begin_epoch=global_step.eval()/(train_len/batch_size)
            begin_epoch=0
            print 'load',args.pretrain,'global_step:',global_step.eval(),'begin epoch:',begin_epoch
                                                          
        else:
            sess.run(tf.global_variables_initializer())
            begin_epoch=0
        train_weight=np.ones([train_len,core[0],core[1],core[2]])
        train_y=np.ones([train_len,core[0],core[1],core[2]],dtype=bool)
        train_weight_scale=np.ones([train_len])
        val_weight=np.ones([len(validation_data_list),core[0],core[1],core[2]])
        f_measure = 0
        neg_finetune=0
        
        
        weight_pretune = 1
        if weight_pretune == 1:
            logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+']  Begining weight pretune!')
            pos_weight_sum, neg_weight_sum = 0, 0
            for iters in range(0,train_len/batch_size):
                #print iters
                if args.boosting_mode == 'SampleAdaBoosting':
                    X,Y,W=get_normalized_data_list(iters,batch_size,train_data,train_data_list,crop_size,border,train_weight,0,neg_finetune,train_weight_scale,train_y=train_y)
                else:
                    X,Y,W=get_normalized_data_list(iters,batch_size,train_data,train_data_list,crop_size,border,train_weight,0,neg_finetune,train_y=train_y)

                pos_weight = np.sum(np.array(Y)*np.array(W))
                pos_weight_sum += pos_weight
                #neg_weight_sum+=np.sum((1-np.array(Y))*np.array(W))
                neg_weight_sum += np.sum(np.array(W)) - pos_weight
            neg_finetune=pos_weight_sum/neg_weight_sum
            logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+']  Neg_finetune:%f'%neg_finetune)
            
            
        ## training
        for epoch in range(begin_epoch,hm_epochs):
            epoch_loss,accuracy_num = 0,0
            epoch_path=savePath+'epoch%d/'% (epoch+1)
            if not os.path.exists(epoch_path):
                os.makedirs(epoch_path)
            with open(epoch_path+'predication.csv', 'wb') as csvfile:
                spamwriter = csv.writer(csvfile)
                spamwriter.writerow(['id', 'cancer','pred'])
                y_pred,y_true = [],[]
                TP_sum,FP_sum,P_sum = 0,0,0
                pos_weight_sum, neg_weight_sum = 0, 0
                FP_rate,FN_rate=0,0
                for iters in range(0,train_len/batch_size):
                    #logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'     data loading')
                    if args.boosting_mode == 'SampleAdaBoosting':
                        X,Y,W=get_normalized_data_list(iters,batch_size,train_data,train_data_list,crop_size,border,train_weight,epoch+weight_pretune,neg_finetune,train_weight_scale,train_y=train_y)
                    else:
                        X,Y,W=get_normalized_data_list(iters,batch_size,train_data,train_data_list,crop_size,border,train_weight,epoch+weight_pretune,neg_finetune,train_y=train_y)
                    #logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'     data got')
                    _, c, a, pred,conf = sess.run([optimizer, loss,accuracy,prediction,confidence], feed_dict={x: X, y: Y, w:W})
                    #logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'     data run')
                    
                    y_pred=np.append(y_pred,conf>0.5)
                    y_true=np.append(y_true,Y)
                    #for i in range(batch_size):
                    #    conf_list.append([conf[i],Y[i],iters*batch_size+i])
                        #spamwriter.writerow([train_data[iters*batch_size+i,2],Y[i],conf[i]])
                    #logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'     Detecting')
                    if args.boosting_mode == 'SampleAdaBoosting':
                        TP,FP,P,train_weight_scale[iters*batch_size:(iters+1)*batch_size] = pos_detect(batch_size,train_data_list[iters*batch_size:(iters+1)*batch_size],conf,Y,weight_scale=train_weight_scale[iters*batch_size:(iters+1)*batch_size],boosting_ratio=1 +(boosting_ratio - 1) * f_measure)
                    elif args.boosting_mode == 'MergeBoosting':
                        TP,FP,P,train_weight[iters*batch_size:(iters+1)*batch_size] = pos_detect_MergeBoosting(batch_size,train_data_list[iters*batch_size:(iters+1)*batch_size],conf,Y,train_weight[iters*batch_size:(iters+1)*batch_size],boosting_ratio=boosting_ratio)#1 +(boosting_ratio - 1) * f_measure)
                    elif args.boosting_mode == 'MergeBoosting_new':
                        TP,FP,P,train_weight[iters*batch_size:(iters+1)*batch_size] = pos_detect_MergeBoosting_new(batch_size,train_data_list[iters*batch_size:(iters+1)*batch_size],conf,Y,train_weight[iters*batch_size:(iters+1)*batch_size],FP_rate,FN_rate,boosting_ratio=1 +(boosting_ratio - 1) * f_measure)
                    else:
                        TP,FP,P = pos_detect(batch_size,train_data_list[iters*batch_size:(iters+1)*batch_size],conf,Y)

                    if args.boosting_mode == 'AdaBoosting':
                        train_weight[iters*batch_size:(iters+1)*batch_size] = Adaboosting(conf,Y,W,boosting_ratio=1 +(boosting_ratio - 1) * f_measure,thresh=0.5)
                    TP_sum+=TP
                    FP_sum+=FP
                    P_sum+=P
                    pos_weight_sum+=np.sum(np.array(Y)*np.array(train_weight[iters*batch_size:(iters+1)*batch_size]))
                    neg_weight_sum+=np.sum((1-np.array(Y))*np.array(train_weight[iters*batch_size:(iters+1)*batch_size]))
                    epoch_loss += c
                    accuracy_num += a/batch_size/core[0]/core[1]/core[2]
						
						## log info
                    if (iters+1) % display == 0:
                        precision=precision_score(y_true, y_pred)
                        recall=recall_score(y_true, y_pred)
                        logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+']  TRAINING:Epoch:%d;Display:%d;Loss:%.4f;accuarcy:%.4f;recall:%.4f;precision:%.4f;TP:%.4f;FP:%.4f;lr:'%(epoch+1, iters / display + 1,epoch_loss / display,accuracy_num / display,recall,precision,float(TP_sum)/P_sum,float(FP_sum)/P_sum)+str(learning_rate.eval()))

                    if iters+1==train_len/batch_size:
                        precision=precision_score(y_true, y_pred)
                        recall=recall_score(y_true, y_pred)
                        logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+']  TRAINING:Epoch:%d;Display:%d;Loss:%.4f;accuarcy:%.4f;recall:%.4f;precision:%.4f;TP:%.4f;FP:%.4f;lr:'%(epoch+1, iters / display + 1,epoch_loss / display,accuracy_num / display,recall,precision,float(TP_sum)/P_sum,float(FP_sum)/P_sum)+str(learning_rate.eval()))
                        y_pred,y_true = [],[]
                        epoch_loss,accuracy_num,precision,recall = 0,0,0,0
                        FP_rate = float(FP_sum)/ P_sum
                        FN_rate = 1 - float(TP_sum)/ P_sum
                        TP_sum,FP_sum,P_sum = 0,0,0
                        
                try:
                    neg_finetune=pos_weight_sum/neg_weight_sum
                except:
                    neg_finetune=0
                
                logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+']TRAINING:Epoch:%d;neg_finetune:%f'%(epoch+1,neg_finetune))
    			## Validation
                num=len(validation_data_list)/batch_size
                y_pred,y_true = [],[]
                epoch_loss,accuracy_num = 0,0
                TP_sum,FP_sum,P_sum = 0,0,0
                spamwriter.writerow([' ', ' ', ' '])
                spamwriter.writerow(['id', 'cancer','pred'])
                for iters in range(0,len(validation_data_list)/batch_size):
                    try:
                        X,Y,W=get_normalized_data_list(iters,batch_size,validation_data,validation_data_list,crop_size,border,val_weight,0)
                        c, a, pred, conf = sess.run([loss,accuracy,prediction,confidence], feed_dict={x: X, y: Y, w:W})
                        
                        y_pred=np.append(y_pred,conf>0.5)
                        y_true=np.append(y_true,Y)
                        #for i in range(batch_size):
                        #    conf_list.append([conf[i],Y[i],iters*batch_size+i])
                        TP,FP,P=pos_detect(batch_size, validation_data_list[iters*batch_size:(iters+1)*batch_size], conf, Y)
                        TP_sum+=TP
                        FP_sum+=FP
                        P_sum+=P
                        epoch_loss += c
                        accuracy_num += a/batch_size/core[0]/core[1]/core[2]
                        
                    except Exception as e:
                        logging.info(('Warning: Test Exception:',str(e)))
                        pass
                precision=precision_score(y_true, y_pred)
                recall=recall_score(y_true, y_pred)     
                try:
                    f_measure = 2 / ((1 / precision) + (1 / recall))
                except:
                    f_measure = 0
                #print f_measure
                if P_sum!=0:
                    logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+']  VALIDATION: Epoch:%d;Loss:%.4f;accuarcy:%.4f;recall:%.4f;precision:%.4f,TP:%.4f;NP:%.4f;'%(epoch+1,epoch_loss / num,accuracy_num / num,recall,precision,float(TP_sum)/P_sum,float(FP_sum)/P_sum))
                #logging.info(str(pos_rate_list))
                #save
                if (epoch+1) % args.save_every ==0:
                    path=epoch_path+'model_%d.ckpt'% (iters/save_every+1)
                    saver.save(sess,path)
                    logging.info( '['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+']  SAVING: save as'+path)
			
        epoch_path=savePath+'final/'
        if not os.path.exists(epoch_path):
                os.makedirs(epoch_path)	
        path=epoch_path+'model_final.ckpt'
        saver.save(sess,path)
        logging.info( '['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'] save as'+path)

# Run this locally:

train_neural_network(x)
