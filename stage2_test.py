import argparse
import csv
import tensorflow as tf
import numpy as np
import data_aug

## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='nopad28_bn_mini2', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--snapshot', default='snapshot/stage2_nopad28_bn_mini2/final/model_final.ckpt', type=str)
parser.add_argument('--validate_size', default=1024, type=int)
parser.add_argument('--transpose', default=False, type=bool)
parser.add_argument('--test', default=False, type=bool)
#parser.add_argument('--data_path', default='data/data_norm0.npy', type=str)
parser.add_argument('--data_path', default='data/data_stage2_test.npy', type=str)
args = parser.parse_args()

test_aug_params = {
	'crop_size': [64,64,64],
	'min_border': [8,8,8],
	'sharpen': False,
	'blur_sigma': 0,
	'noise_max_amplitude': 0,
	'flip': False,
	'rot': False,
	}
if args.model=='res3d20':
	import models.res3d20 as model
elif args.model=='nopad28_bn':
	import models.nopad28_bn as model
elif args.model=='nopad28_bn_mini':
	import models.nopad28_bn_mini as model
elif args.model=='nopad28_bn_mini2':
	import models.nopad28_bn_mini2 as model
else:
	import models.res3d20 as model





## args
batch_size = args.batch_size # batch_size <=4 , for memory constrain

## Net setting	
x = tf.placeholder('float')
#prediction = tf.reduce_max(model.CNN(x),[1,2,3,4])
prediction = tf.reduce_max(model.CNN_test(x),[4])
#a,b=model.CNN(x)
#prediction = tf.reduce_max(a,[1,2,3,4])
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


def get_normalized_data(iters,batch_size,data,aug_params):
    if args.transpose:
        X=[np.transpose(data_aug.data_aug(data[i,0],aug_params).astype(float)/128-1,[2,1,0]) for i in range(iters*batch_size,(iters+1)*batch_size)]
    else:
        X=[data_aug.data_aug(data[i,0],aug_params).astype(float)/128-1 for i in range(iters*batch_size,(iters+1)*batch_size)]
        
    X=np.reshape(X,[-1,aug_params['crop_size'][0],aug_params['crop_size'][1],aug_params['crop_size'][2],1])
    return X
				

def predict(X):    
    return sess.run(confidence, feed_dict={x: X})	   
            
# Run this locally:
if __name__ == "__main__":
    ## load data    
    data_path=args.data_path
    data = np.load(data_path)  
    if args.test :
        train_data = data[:-args.validate_size]
        validation_data = data[-args.validate_size:]
        train_len=train_data.shape[0]
        train_positive_rate=np.sum(train_data[:,1].astype(float))/train_data.shape[0]
        val_positive_rate=np.sum(validation_data[:,1].astype(float))/validation_data.shape[0]
        X=get_normalized_data(0,batch_size,validation_data,test_aug_params)
        Y=predict(X)
        print np.reshape(Y,[batch_size])
        print validation_data[0:64,1]
    else:
        data_len=data.shape[0]
        with open('predication.csv', 'wb') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(['seriesuid','coordX','coordY','coordZ','probability','label'])
            for i in range(data_len/batch_size):
                X=get_normalized_data(i,batch_size,data,test_aug_params)
                Y=predict(X)
                other=[data[i,1:] for i in range(i*batch_size,(i+1)*batch_size)]
                for j in range(batch_size):
                    spamwriter.writerow([other[j][1],other[j][2][0],other[j][2][1],other[j][2][2],Y[j],other[j][0]])
  
            print 'done'
            
    sess.close()
