
import pandas as pd
from glob import glob
from joblib import Parallel, delayed
import image


mode='train'

def read(imagePath,cads):
    image.read_mhd(imagePath,cads,mode,norm_flag=True,mask_flag=True,save=True)



imagePaths = glob('../%s/*/*.mhd' % mode)
cads = pd.read_csv('../csv/%s/annotations.csv' % mode)


Parallel(n_jobs=8)(delayed(read)(imagePath,cads) for imagePath in imagePaths)   