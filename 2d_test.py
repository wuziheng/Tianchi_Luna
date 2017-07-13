import argparse
import tensorflow as tf
import numpy as np
import data_aug
from FROC_test import FROC
import data_aug

IMAGE_SIZE=64
PHASE = 'val'
#PHASE = 'test'
EPOCH = 15
neg_rate_list = [0.125,0.25,0.5,1,2,4,8]
TARGET_SPACING = [0.70216,0.70216,1.1163]

## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='resnet2d', type=str)
parser.add_argument('--batch_size', default=64, type=int)
#parser.add_argument('--snapshot', default='snapshot/save_resnet2d_2017-07-07-18:34:22/epoch2/model_1.ckpt', type=str)
parser.add_argument('--snapshot', default='snapshot/save_resnet2d_2017-07-09-21:21:13/epoch%d/model_1.ckpt'%(EPOCH), type=str)
parser.add_argument('--transpose', default=False, type=bool)
if PHASE == 'test':
    parser.add_argument('--data_path', default='data/data_%s_2d_Z.npy'%PHASE, type=str)
    #parser.add_argument('--data_path', default='data/data_%s_2d_masklarge.npy'%PHASE, type=str)
if PHASE == 'val':
    parser.add_argument('--data_path', default='data/data_%s_2d_Z.npy'%PHASE, type=str)
    #parser.add_argument('--data_path', default='data/data_classify_val_large_mask.npy', type=str)
args = parser.parse_args()

test_aug_params = {
	'crop_size': [IMAGE_SIZE,IMAGE_SIZE,1],
	'min_border': [8,8,0],
	'zoom':1.0,
	'random_crop':False,
	'sharpen': False,
	'blur_sigma': 0,
	'noise_max_amplitude': 0,
	'flip': False,
	'rot': False,
	}


if args.model=='resnet2d':
	import models.resnet2d_64_v2 as model

else:
	import models.res3d20 as model

d = {}
if PHASE == 'test':
    origin  =  np.load('test2/test2_origin.npy')
    for o in origin:
        d[o[1]] = o[0]
    print d

## args
batch_size = args.batch_size # batch_size <=4 , for memory constrain

## Net setting	
x = tf.placeholder('float',shape=[batch_size,IMAGE_SIZE,IMAGE_SIZE,1])

prediction_original=model.inference_small(x,is_training=False)
prediction = tf.reduce_max(prediction_original,[1,2,3])        
confidence=tf.nn.sigmoid(prediction)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver(max_to_keep=10000)
#with tf.Session(config=config) as sess:
# Init
try:
	sess=tf.Session(config=config)
	saver.restore(sess,args.snapshot)
	print 'load',args.snapshot
except:
    print 'wrong'


def get_normalized_data(iters,batch_size,data,aug_params=test_aug_params):
    X=[data_aug.data_aug_2d(data[i,0],aug_params).astype(float)/128-1 for i in range(iters*batch_size,(iters+1)*batch_size)]
    Y=[float(data[i,1]) for i in range(iters*batch_size,(iters+1)*batch_size)]
    X=np.reshape(X,[batch_size,IMAGE_SIZE,IMAGE_SIZE,1])
    return X,Y
				

def predict(X):
    return sess.run([confidence], feed_dict={x: X})	   
            
import csv
# Run this locally:
if __name__ == "__main__":
    ## load data    
    data_path=args.data_path
    data = np.load(data_path) 
    print data.shape
    validation_scans_num=len(set(data[:,2]))
    validation_pos_num=270
    #validation_pos_num=sum(data[:,1].astype(float))
    print 'validation_scans_num: %s, validation_pos_num: %s'%(validation_scans_num,validation_pos_num)
    conf_list = []
    with open('2d_predication.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['seriesuid', 'coordX','coordY','coordZ','probability'])
             
        for j in range(len(data)/batch_size): 
            X,Y = get_normalized_data(j,batch_size,data)
            conf = predict(X)[0]
            if PHASE =='val':
                for i in range(batch_size):
                    conf_list.append([conf[i],Y[i],j*batch_size+i])
            if PHASE == 'test':
                for i in range(batch_size):
                    coord = data[j*batch_size+i,3]*TARGET_SPACING+d[data[j*batch_size+i,2]] 
                    print data[j*batch_size+i,2],conf[i],coord
                    #if conf[i] > 0.00002:
                    spamwriter.writerow([data[j*batch_size+i,2],coord[0],coord[1],coord[2],conf[i]])

    if PHASE == 'val':
        pos_rate_list=FROC(conf_list,neg_rate_list,validation_scans_num,validation_pos_num)
        print pos_rate_list
        print 'froc8 : ',sum(pos_rate_list)/7
    
    sess.close()
