import sys
sys.path.append('..')
import os
import time
import logging
import numpy as np
from glob import glob
import util.image as image

## generate config
phase = 'train'
image_type = '3d'
image_size = 32
model_name = '3dcnn'
##

def generator(  PHASE=phase,
                IMAGE_TYPE=image_type,
                IMAGE_SIZE=image_size,
                MODEL_NAME=model_name):
    ROOT_PATH = os.path.abspath('..')
    DATA_PATH = '%s/Cache/net1_out_%s/'%(ROOT_PATH,PHASE)
    LOG_PATH = '%s/generator/log/'%ROOT_PATH
    SAVE_PATH = '%s/data/'%ROOT_PATH
    
    ## path config
    if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)	
    if not os.path.exists(LOG_PATH):
            os.mkdir(LOG_PATH)	
    
    ## log config	
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y %H:%M:%S',
                    filename= LOG_PATH+time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())+'.log',
                    filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    ## 
    
    
    logging.info('DATA_PATH: %s'%DATA_PATH)
    logging.info('DATA_PHASE: %s'%PHASE) 
    logging.info('IMAGE_TYPE: %s'%IMAGE_TYPE) 
    logging.info('IMAGE_SIZE: %d'%IMAGE_SIZE)
    logging.info('MODEL_NAME: %s'%MODEL_NAME)
    file_list = glob('%s/L*.npy'%DATA_PATH)
    logging.info( 'START_TIME: ['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'], TOTAL_CT: %d'%len(file_list))
    
    x = []
    total = 0
    pos_num = 0
    for f in file_list:
        try:
        #if 1:
            fname = f.split('/')[-1]
            f_total = 0
            f_pos_num = 0
            img = np.load('{0}/Cache/normed_{1}/{2}'.format(ROOT_PATH,PHASE,fname))[0]
            coords = np.load(f) 
            coords_aug = list(coords)
            for i in range(coords.shape[0]):
                label = 0
                crop_success_flag = False
                name = coords[i][1]
                center = np.array(coords[i][0],dtype=int)
                nodes = coords[i][2]
                for node in nodes:
                    r = node[1]/2
                    c = node[0]
                    if np.max(np.abs(c-center)/r)<1:
                        label = 1
    
                t = image.crop(img,center,IMAGE_SIZE/2)
                crop_success_flag = t[0]
    
                if crop_success_flag == True:
                    total+=1
                    f_total+=1
                    if IMAGE_TYPE=='3d':
                        x.append([t[1],label,name])
                    if IMAGE_TYPE=='2d':
                        x.append([t[1][:,:,IMAGE_SIZE/2],label,name])
                    if label == 1:
                        f_pos_num+=1
                        pos_num+=1
            print '{0} generate total {1:<3} candidata,  {2} pos at'.format(fname,f_total,f_pos_num),time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        except:
            logging.info(('Warning: %s generate Exception:'%fname))
            pass
             
    np.save('{0}/data/data_{1}_{2}_{3}.npy'.format(ROOT_PATH,PHASE,MODEL_NAME,IMAGE_SIZE),x)
    logging.info( 'END_TIME: ['+time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+'], data_num: %d, pos_num: %d'%(total,pos_num))
    
    
#generator()
