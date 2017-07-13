import argparse
import csv
import tensorflow as tf
import numpy as np

import glob
TARGET_SPACING = [0.70216,0.70216,1.1163]
Z_RATE=TARGET_SPACING[2]/TARGET_SPACING[1]
## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='resnet_seg_fat', type=str)
parser.add_argument('--snapshot', default='snapshot/resnet_seg_fat/final/model_final.ckpt', type=str)
parser.add_argument('--data_path', default='data/data_stage2_test.npy', type=str)
parser.add_argument('--batch_size', default=8, type=int)
args = parser.parse_args()
if args.model=='seg':
	import models.seg as model
elif args.model=='resnet':
	import models.resnet as model
elif args.model=='resnet_seg_fat':
	import models.resnet_seg_fat as model
else:
	import models.seg as model


## Net setting	
border=np.array([34,34,34])
crop_size=np.array([120,120,120])
core=crop_size-2*border
x = tf.placeholder('float',shape=[args.batch_size,crop_size[0],crop_size[1],crop_size[2],1])
x = tf.placeholder('float')
prediction=model.tf.reduce_max(model.inference_small(x,is_training=False),4)
confidence=tf.nn.sigmoid(prediction)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver(max_to_keep=10000)
# Init
try:
	sess=tf.Session(config=config)
	saver.restore(sess,args.snapshot)
	print 'load',args.snapshot
except:
    print 'wrong'

def get_patch(img,crop_size,begin_coord):    
    end_coord=begin_coord+crop_size
    x=np.zeros([crop_size[0],crop_size[1],crop_size[2]])
    real_begin_coord = np.maximum(begin_coord,0)
    real_end_coord = np.minimum(end_coord,img.shape)
    x_begin_coord = real_begin_coord - begin_coord
    x_end_coord = real_end_coord - real_begin_coord + x_begin_coord
    x[x_begin_coord[0]:x_end_coord[0],x_begin_coord[1]:x_end_coord[1],x_begin_coord[2]:x_end_coord[2]] = img[real_begin_coord[0]:real_end_coord[0],real_begin_coord[1]:real_end_coord[1],real_begin_coord[2]:real_end_coord[2]]
    return x

def predict(X):
    
    return sess.run(confidence, feed_dict={x: X})	 

def predict_full(img):
    Y=np.zeros(np.array(img.shape))
    num_patch=np.array(Y.shape)/core
    X=[]
    X_coord_list=[]
    for i in range(num_patch[0]):
        for j in range(num_patch[1]):
            for k in range(num_patch[2]):
                begin_coord=np.array([i,j,k])*core-border
                patch=get_patch(img,crop_size,begin_coord)    
                X.append(patch)
                X_coord_list.append(np.array([i,j,k]))
                if len(X)==args.batch_size:
                    X=np.reshape(X,[args.batch_size,crop_size[0],crop_size[1],crop_size[2],1])
                    X=(X.astype(float)-128)/2
                    print X.shape
                    y=sess.run(confidence, feed_dict={x: X})
                    for index in range(args.batch_size):
                        coord=X_coord_list[index]
                        copy_shape=np.minimum(np.array(Y.shape),(coord+1)*core)-coord*core
                        Y[coord[0]*core[0]:coord[0]*core[0]+copy_shape[0],coord[1]*core[1]:coord[1]*core[1]+copy_shape[1],coord[2]*core[2]:coord[2]*core[2]+copy_shape[2]] = y[index,0:copy_shape[0],0:copy_shape[1],0:copy_shape[2]]
                    X=[]
                    X_coord_list=[]
    

    if X==[]:
        return Y
    else:
        while len(X)<args.batch_size:
            X.append(patch)
        
        X=np.reshape(X,[args.batch_size,crop_size[0],crop_size[1],crop_size[2],1])
        X=(X.astype(float)-128)/2
        y=sess.run(confidence, feed_dict={x: X})
        for index in range(len(X_coord_list)):
            coord=X_coord_list[index]
            copy_shape=np.minimum(np.array(Y.shape),(coord+1)*core)-coord*core
            Y[coord[0]*core[0]:coord[0]*core[0]+copy_shape[0],coord[1]*core[1]:coord[1]*core[1]+copy_shape[1],coord[2]*core[2]:coord[2]*core[2]+copy_shape[2]] = y[index,0:copy_shape[0],0:copy_shape[1],0:copy_shape[2]]
        return (Y*255).astype(np.uint8)	   
            

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

# Run this locally:
if __name__ == "__main__":
            
    mode = 'train'
    imagePaths = glob.glob('/home/lijunying/luna17/Cache/normed_%s/*.npy' % mode)
    savedata = []
    i=0
    for path in imagePaths:
        i+=1
        print path,i
        img,imageName,nodes=np.load(path)
        print imageName,img.shape,nodes
        label=get_label(img,imageName,nodes,get_label=True)
        Y=predict_full(img)
        np.save(imageName,Y)
        #savedata.append([Y,imageName,nodes])
        break
    #np.save('tmp.npy',savedata)
        
        
    sess.close()
