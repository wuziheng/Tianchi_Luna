import argparse
import tensorflow as tf
import numpy as np
import time
import pandas as pd
import csv
import os
import random
from sklearn.metrics import precision_score,recall_score
import logging
import data_aug
from FROC_test import FROC

## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='resnet2d_34', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--display', default=300, type=int)
parser.add_argument('--save_every', default=5, type=int)
parser.add_argument('--epoch', default=40, type=int)
parser.add_argument('--decay_epoch', default=17, type=int)
parser.add_argument('--begin_step', default=0, type=int)
parser.add_argument('--learning_rate', default=2e-3, type=float)
parser.add_argument('--boosting_ratio', default=2, type=float)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--shuffle', default=False, type=bool)    
parser.add_argument('--show', default=False, type=bool)
parser.add_argument('--train_data_path', default='data/data_classify_train_v3.npy', type=str)
parser.add_argument('--val_data_path', default='data/data_classify_val_v3.npy', type=str)
parser.add_argument('--boosting_mode', default='NoDisAddBoosting', type=str)
parser.add_argument('--gpu', default=3, type=int)
args = parser.parse_args()
train_aug_params = {
	'crop_size': [64,64,2],
	'min_border': [0,0,0],
	'zoom':1.0,
	'random_crop':True,
	'sharpen': False,
	'blur_sigma': 0,
	'noise_max_amplitude': 0,
	'flip': False,
	'rot': False,
	}
test_aug_params = {
	'crop_size': [64,64,2],
	'min_border': [8,8,0],
	'zoom':1.0,
	'random_crop':False,
	'sharpen': False,
	'blur_sigma': 0,
	'noise_max_amplitude': 0,
	'flip': False,
	'rot': False,
	}
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
if args.model=='resnet2d':
	import models.resnet2d as model
elif args.model=='resnet2d_101':
	import models.resnet2d_101 as model
elif args.model=='resnet2d_50':
	import models.resnet2d_50 as model
elif args.model=='resnet2d_34':
	import models.resnet2d_34 as model
elif args.model=='neg_fuse':
	import models.neg_fuse as model
elif args.model=='neg_ori':
	import models.neg_ori as model
else:
	print 'no corresponding model'
	import models.res3d20 as model

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
formatter = logging.Formatter('%(name)-6s: %(levelname)-6s %(message)s')
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

## load data    
logging.info( 'loading data from '+ args.train_data_path)
train_data = np.load(args.train_data_path)
logging.info( 'loading data from '+ args.val_data_path)
validation_data = np.load(args.val_data_path)
#train_info = pd.read_csv('input/train_data.csv')
if args.shuffle:
	index=range(train_data.shape[0])
	random.shuffle(index)
	train_data=train_data[index]

train_len=(train_data.shape[0]/batch_size)*batch_size# make it times of batch_size
train_data=train_data[0:train_len]    
      
train_positive_rate=np.sum(train_data[:,1].astype(float))/train_data.shape[0]
logging.info( 'train_data.shape:'+str(train_data.shape)+' positive rate:' + str(train_positive_rate))
val_positive_rate=np.sum(validation_data[:,1].astype(float))/validation_data.shape[0]
logging.info( 'validation_data.shape:'+str(validation_data.shape)+' positive rate:' + str(val_positive_rate))

train_scans_num=len(set(train_data[:,2]))
validation_scans_num=len(set(validation_data[:,2]))
train_pos_num=sum(train_data[:,1].astype(float))
validation_pos_num=sum(validation_data[:,1].astype(float))
neg_rate_list=np.array([0.125,0.25,0.5,1,2,4,8])
logging.info( 'train_scans: %d;validation_scans: %d;train_pos: %d;validation_pos: %d'%(train_scans_num,validation_scans_num,train_pos_num,validation_pos_num))

crop_size = np.array(train_aug_params['crop_size'])
x = tf.placeholder('float',shape=[batch_size,crop_size[0],crop_size[1],crop_size[2]])
y = tf.placeholder('float')
w = tf.placeholder('float')

def weight_refine_LiBoosting(weight,conf_list,positive_rate,pos_num):
    
    if boosting_ratio==0:
        return weight
    
    conf_list.sort()
    conf_list.reverse()
    num=len(conf_list)
    pos=0
    neg=0
    pos_weight_sum=0
    neg_weight_sum=0
    neg_num=num-pos_num
    for i in range(num):
        if conf_list[i][1]:
            pos+=1
            weight[conf_list[i][2]] = min(float(neg) / (neg_num) , 1) * boosting_ratio + 1
            pos_weight_sum += weight[conf_list[i][2]]
        else:
            neg+=1
            weight[conf_list[i][2]] = float(pos_num-pos) / pos_num * boosting_ratio +1
            neg_weight_sum += weight[conf_list[i][2]]
                                                                   
                                       
    for i in range(num):
        if conf_list[i][1]:
            weight[conf_list[i][2]] /= (pos_weight_sum/pos_num)/((1-positive_rate)/positive_rate)
        else:
            weight[conf_list[i][2]] /= (neg_weight_sum/neg_num)
    return weight

def weight_refine_AdaBoosting(weight,conf_list,positive_rate,pos_num,max_mul):
    
    if boosting_ratio==0:
        return weight
    conf_list.sort()
    conf_list.reverse()
    num=len(conf_list)
    pos=0
    neg=0
    pos_weight_sum=0
    neg_weight_sum=0
    neg_num=num-pos_num
    for i in range(num):
        if not conf_list[i][1]:
            top_pos_num=i+1
            break
    for i in range(num):
        if conf_list[num-i-1][1]:
            mid_neg_num=num-i-1-pos_num
            break        
    for i in range(num):
        if conf_list[i][1]:
            pos+=1
            if neg>0:
                weight[conf_list[i][2]] = weight[conf_list[i][2]] * (min(float(neg) / mid_neg_num , 1) * (max_mul-1) + 1)
            else:
                weight[conf_list[i][2]] = weight[conf_list[i][2]] / (float(top_pos_num-pos) / (top_pos_num) * (max_mul-1) + 1)
            pos_weight_sum += weight[conf_list[i][2]]
        else:
            neg+=1
            if pos< pos_num:
                weight[conf_list[i][2]] = weight[conf_list[i][2]] * (float(pos_num-pos) / (pos_num-top_pos_num) * (max_mul-1) + 1)
                wrong_neg=neg
            else:
                weight[conf_list[i][2]] = weight[conf_list[i][2]] / (float(neg-wrong_neg) / (neg_num-wrong_neg) * (max_mul-1) + 1)
            neg_weight_sum += weight[conf_list[i][2]]
                                                                                                       
    for i in range(num):
        if conf_list[i][1]:
            weight[conf_list[i][2]] /= (pos_weight_sum/pos_num)/((1-positive_rate)/positive_rate)
        else:
            weight[conf_list[i][2]] /= (neg_weight_sum/neg_num)
    
    print 'adaboost: max_mul:',max_mul,',max:',np.max(weight),',min:',np.min(weight)
    return weight

def weight_refine_AddBoosting(weight,conf_list,positive_rate,pos_num,max_mul):
    
    if boosting_ratio==0:
        return weight
    conf_list.sort()
    conf_list.reverse()
    num=len(conf_list)
    pos=0
    neg=0
    pos_weight_sum=0
    neg_weight_sum=0
    neg_num=num-pos_num
    for i in range(num):
        if not conf_list[i][1]:
            top_pos_num=i+1
            break
    for i in range(num):
        if conf_list[num-i-1][1]:
            mid_neg_num=num-i-1-pos_num
            break        
    weight_sum_old = np.sum(weight)
    for i in range(num):
        if conf_list[i][1]:
            pos+=1
            if neg>0:
                weight[conf_list[i][2]] += float(neg) / mid_neg_num  * (max_mul-1) / positive_rate

            pos_weight_sum += weight[conf_list[i][2]]
        else:
            neg+=1
            if pos< pos_num:
                weight[conf_list[i][2]] += float(pos_num-pos) / (pos_num-top_pos_num) * (max_mul-1) / positive_rate
            neg_weight_sum += weight[conf_list[i][2]]
    weight_sum_new = np.sum(weight)  
    if 0:                                                                                                 
        for i in range(num):
            if conf_list[i][1]:
                weight[conf_list[i][2]] /= (pos_weight_sum/pos_num)/((1-positive_rate)/positive_rate)
            else:
                weight[conf_list[i][2]] /= (neg_weight_sum/neg_num)
    else:
        weight *= weight_sum_old/weight_sum_new
    
    print 'addboost: max_mul:',max_mul,',max:',np.max(weight),',min:',np.min(weight)
    return weight

def weight_refine_DisAddBoosting(weight,conf_list,neg_rate_list,positive_rate,pos_num,scans_num,max_mul):
    
    if boosting_ratio==0:
        return weight
    conf_list.sort()
    conf_list.reverse()
    pos,neg,index = 0,0,0
    pos_num_list,neg_add_list = [],[]
    weight_sum_old = np.sum(weight)
    neg_rate_list = np.array(neg_rate_list) * 3
    ## get pos_num_list
    for i in range(len(conf_list)):
        if conf_list[i][1]:
            pos+=1
        else:
            neg+=1
            if neg==1:
                top_pos=pos
        if neg>=scans_num * neg_rate_list[index]:
            pos_num_list.append(pos)
            if index >= len(neg_rate_list)-1:       
                break
            else:
                index+=1
    ## get neg_add_list
    for index in range(len(neg_rate_list)):
        if index ==0:
            add = (pos_num_list[index] - top_pos) / (scans_num * neg_rate_list[index])
        else:
            add = (pos_num_list[index] - pos_num_list[index - 1]) / (scans_num * (neg_rate_list[index] - neg_rate_list[index - 1]))
        neg_add_list.append(add)
        
    for index in range(1,len(neg_add_list)):
        neg_add_list[-index-1]+=neg_add_list[-index]
        
    pos,neg,index = 0,0,0
    for i in range(len(conf_list)):
        if conf_list[i][1]:
            pos+=1
        else:
            neg+=1
        if neg > 0:
            if conf_list[i][1]:
                weight[conf_list[i][2]] += (index + 1)*(max_mul-1)
            elif index < len(neg_rate_list):
                weight[conf_list[i][2]] += (neg_add_list[index])*(max_mul-1)
                if neg>=scans_num*neg_rate_list[index]:
                        index+=1

    weight_sum_new = np.sum(weight)  
    weight *= weight_sum_old/weight_sum_new
    pos_w,neg_w=0,0
    for i in range(len(conf_list)):
            if conf_list[i][1]:
                pos_w += weight[conf_list[i][2]]
            else:
                neg_w += weight[conf_list[i][2]] 
    print 'addboost: max_add:',max_mul -1,',max:',np.max(weight),',min:',np.min(weight),',pos/neg:',pos_w/neg_w,',neg_add_list:',neg_add_list
    return weight



def wzh_aug(img,istrain):
    rangex=img.shape[0]-64
    istrain = False
    x = random.randint(0,rangex) if istrain else rangex/2
    y = random.randint(0,rangex) if istrain else rangex/2
    
    return img[x:x+64,y:y+64,:]

def get_normalized_data(iters,batch_size,data,aug_params,weight,istrain=1):

    #X=[wzh_aug(data[i,0],istrain).astype(float)/128-1 for i in range(iters*batch_size,(iters+1)*batch_size)]
    X=[data_aug.data_aug_2d(data[i,0],aug_params).astype(float) for i in range(iters*batch_size,(iters+1)*batch_size)]
    
    Y=[float(data[i,1]) for i in range(iters*batch_size,(iters+1)*batch_size)]
    #X=np.reshape(X,[-1,aug_params['crop_size'][0],aug_params['crop_size'][1],aug_params['crop_size'][2],1])
    X=np.reshape(X,[-1,crop_size[0],crop_size[1],crop_size[2]])
    X[:,:,:,0]=X[:,:,:,0]/128 -1
    W=[weight[i].astype(float) for i in range(iters*batch_size,(iters+1)*batch_size)]       
    return X,Y,W

def train_neural_network(x):
    
    # Setting

    prediction_original=model.inference_small(x,is_training=True)
    prediction = tf.reduce_mean(prediction_original,[1,2,3])
    loss = tf.reduce_mean( w*tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=prediction))
    confidence=tf.nn.sigmoid(prediction)
    correct = tf.equal(tf.greater(confidence, 0.5),tf.greater(y, 0))
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
            
        ## training
        train_weight = train_data[:,1].astype(float)*((1-train_positive_rate)/train_positive_rate-1)+1
        val_weight = validation_data[:,1].astype(float)*((1-val_positive_rate)/val_positive_rate-1)+1
        f_measure = 0
        for epoch in range(begin_epoch,hm_epochs):
            epoch_loss = 0
            accuracy_num=0
            y_pred=[]
            y_true=[]
            conf_list=[]
            epoch_path=savePath+'epoch%d/'% (epoch+1)
            if not os.path.exists(epoch_path):
                os.makedirs(epoch_path)
            with open(epoch_path+'predication.csv', 'wb') as csvfile:
                spamwriter = csv.writer(csvfile)
                spamwriter.writerow(['id', 'cancer','pred'])
                for iters in range(0,train_data.shape[0]/batch_size):
                   
                    X,Y,W=get_normalized_data(iters,batch_size,train_data,train_aug_params,train_weight)                    
                    _, c, a, pred,conf = sess.run([optimizer, loss,accuracy,prediction,confidence], feed_dict={x: X, y: Y, w:W})
                        
                    y_pred=np.append(y_pred,conf>0.5)
                    y_true=np.append(y_true,Y)
                    for i in range(batch_size):
                        conf_list.append([conf[i],Y[i],iters*batch_size+i])
                        spamwriter.writerow([train_data[iters*batch_size+i,2],Y[i],conf[i]])
                    epoch_loss += c
                    accuracy_num += a/batch_size
						
						## log info
                    if (iters+1) % display == 0 or iters == train_data.shape[0]/batch_size -1:
                        precision=precision_score(y_true, y_pred)
                        recall=recall_score(y_true, y_pred)
                        if args.show:
                            logging.info(str(y_true)+str(y_pred))##### print
                        logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+']TRAINING:Epoch:%d;Display:%d;Loss:%.4f;accuarcy:%.4f;recall:%.4f;precision:%.4f;learning_rate:'%(epoch+1, iters / display + 1,epoch_loss / display,accuracy_num / display,recall,precision)+str(learning_rate.eval()))
                        y_pred=[]
                        y_true=[]
                        epoch_loss=0
                        accuracy_num=0
                        precision=0
                        recall=0
                
                pos_rate_list=FROC(conf_list,neg_rate_list,train_scans_num,train_pos_num)
                s='TRAINING: FROC:['
                for rate in pos_rate_list:
                    s += '%.4f, ' % rate
                s +=']'
                logging.info(s+ ' Mean:%.4f' % np.mean(pos_rate_list))
                max_mul = f_measure * (boosting_ratio - 1) + 1
                if boosting_mode=='LiBoosting':
                    train_weight=weight_refine_LiBoosting(train_weight,conf_list,train_positive_rate,train_pos_num)
                elif boosting_mode=='AdaBoosting':
                    train_weight=weight_refine_AdaBoosting(train_weight,conf_list,train_positive_rate,train_pos_num,max_mul)
                elif boosting_mode=='AddBoosting':
                    train_weight=weight_refine_AddBoosting(train_weight,conf_list,train_positive_rate,train_pos_num,max_mul)
                elif boosting_mode=='DisAddBoosting':
                    train_weight=weight_refine_DisAddBoosting(train_weight,conf_list,neg_rate_list,train_positive_rate,train_pos_num,train_scans_num,max_mul)
                ## Validation
                accuracy_num=0
                num=validation_data.shape[0]/batch_size
                epoch_loss=0
                y_pred=[]
                y_true=[]
                conf_list=[]
                spamwriter.writerow([' ', ' ',' '])
                spamwriter.writerow(['id', 'cancer','pred'])
                for iters in range(0,validation_data.shape[0]/batch_size):                  
                    X,Y,W=get_normalized_data(iters,batch_size,validation_data,test_aug_params,val_weight,0)
                    c, a, pred, conf = sess.run([loss,accuracy,prediction,confidence], feed_dict={x: X, y: Y, w:W})                   
                    y_pred=np.append(y_pred,conf>0.5)
                    y_true=np.append(y_true,Y)
                    for i in range(batch_size):
                        conf_list.append([conf[i],Y[i],iters*batch_size+i])
                        spamwriter.writerow([validation_data[iters*batch_size+i,2],Y[i],conf[i]])
                    epoch_loss += c
                    accuracy_num += a/batch_size

                precision=precision_score(y_true, y_pred)
                recall=recall_score(y_true, y_pred)    
                try:
                    f_measure = 2 / ((1 / precision) + (1 / recall))
                except:
                    f_measure = 0
                pos_rate_list=FROC(conf_list,neg_rate_list,validation_scans_num,validation_pos_num)
                logging.info('['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+']VALIDATION: Epoch:%d;Loss:%.4f;accuarcy:%.4f;recall:%.4f;precision:%.4f'%(epoch+1,epoch_loss / num,accuracy_num / num,recall,precision))
                s='VALIDATION: FROC:['
                for rate in pos_rate_list:
                    s += '%.4f, ' % rate
                s +=']'
                logging.info(s + ' Mean:%.4f' % np.mean(pos_rate_list))
                #save
                if (epoch + 1) % args.save_every==0:
                    path=epoch_path+'model_%d.ckpt'% (iters/save_every+1)
                    saver.save(sess,path)
                    logging.info( '['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'] save as'+path)
			
        epoch_path=savePath+'final/'
        if not os.path.exists(epoch_path):
                os.makedirs(epoch_path)	
        path=epoch_path+'model_final.ckpt'
        saver.save(sess,path)
        logging.info( '['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'] save as'+path)

# Run this locally:

train_neural_network(x)
