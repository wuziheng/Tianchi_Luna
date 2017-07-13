from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
#import cv2
import numpy as np 
import pandas as pd 
import dicom
import os
import scipy.ndimage
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import time
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import SimpleITK as sitk
import time

RESIZE_SPACING = [1, 1, 1]
TARGET_SPACING = [0.70216,0.70216,1.1163]
target = np.array(TARGET_SPACING,dtype=float)

def plot_3d(image, threshold=-300):
# Position the scan upright,
# so the head of the patient would be at the top facing the camera
	p = image.transpose(2, 1, 0)

	verts, faces = measure.marching_cubes(p, threshold)

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
	mesh = Poly3DCollection(verts[faces], alpha=0.70)
	face_color = [0.45, 0.45, 0.75]
	mesh.set_facecolor(face_color)
	ax.add_collection3d(mesh)

	ax.set_xlim(0, p.shape[0])
	ax.set_ylim(0, p.shape[1])
	ax.set_zlim(0, p.shape[2])

	plt.show()

def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    image = np.transpose(sitk.GetArrayFromImage(itkimage))
    origin = np.array(itkimage.GetOrigin())
    spacing = np.array(itkimage.GetSpacing())
    return image, origin, spacing

def world_2_voxel(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord

def voxel_2_world(voxel_coord, origin, spacing):
    stretched_voxel_coord = voxel_coord * spacing
    world_coord = stretched_voxel_coord + origin
    return world_coord


def nms(data,a1=1.05,a2=1,a3=1):
	for i in range(1,data.shape[0]-1):
		for j in range(1,data.shape[1]-1):
			for k in range(1,data.shape[2]-1):
				if data[i][j][k]==np.max(data[i-1:i+2,j-1:j+2,k-1:k+2]):
					data[i][j][k] = a1*data[i][j][k]
					data[i][j][k] = 1 if data[i][j][k]>1 else data[i][j][k]
					
				if data[i][j][k]>=np.mean(data[i-1:i+2,j-1:j+2,k-1:k+2]):
					data[i][j][k] = a2*data[i][j][k]
				else:
					data[i][j][k] = a3*data[i][j][k]
	return data

def norm(image,window=700):
	image=image+600+window/2
	image[image > window]=window
	image[image < 0]=0;
	image=image.astype(float)
	image=image*255/window
	image=image.astype(np.uint8)
	return image

def norm1(image,MIN=-1000,MAX=400):
    image = (image-MIN)/(MAX-MIN)
    image[image>1] = 1
    image[image<0] = 0
    return image

def get_threshold(image,keep_rate=0.02):
    imgcount =np.count_nonzero(image)
    hist = np.histogram(image.flatten(),bins=500)[0]
    #hist = plt.hist(image.flatten(),500)[0]
    hist_idx = -1
    count = hist[hist_idx]
    while count < keep_rate*imgcount:
        hist_idx-=1
        count+=hist[hist_idx]
    thre = 1+0.002*(hist_idx)
    #print 'keep_rate =',keep_rate,'thre =',thre
    return thre

def threshold1(image,keep_rate=0.02):
    thre = get_threshold(image,keep_rate)
    return threshold(image,thre),thre

def threshold(image,threshold=0.978):
	proposals = np.array(np.where(image>threshold)).transpose(1,0)
	return proposals

def generate(img,getmean):
	# print "fuck"
	#plt.subplot(221),plt.imshow(img,'gray')
	
	mean = np.mean(img)
	std = np.std(img)
	img = img-mean
	img = img/std
	#plt.subplot(222),plt.hist(img.flatten(),bins=200)
	
	middle = img[100:400,100:400] 
	mean = np.mean(middle)  
	max0 = np.max(img)
	min0 = np.min(img)
	
	#move the underflow bins
	img[img==max0]=mean
	img[img==min0]=mean
	kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
	centers = sorted(kmeans.cluster_centers_.flatten())
	threshold = np.mean(centers)
	thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
	
	# return thresh_img
	#plt.subplot(222),plt.imshow(thresh_img,'gray')
	
	
	eroded = morphology.erosion(thresh_img,np.ones([4,4]))
	dilation = morphology.dilation(eroded,np.ones([15,15]))
	labels = measure.label(dilation)
	#label_vals = np.unique(labels)
	
	# plt.subplot(223),plt.imshow(labels,'gray')
	
	
	regions = measure.regionprops(labels)
	good_labels = []
	for prop in regions:
	    B = prop.bbox
	    #print B
	    #if B[2]-B[0]<500 and B[3]-B[1]<500 and B[0]>10 and B[2]<img.shape[0]-10:
	    #    good_labels.append(prop.label)
	    if  B[0]>10 and B[2]<img.shape[0]-10:
	        good_labels.append(prop.label)
	

	mask = np.ndarray([img.shape[0],img.shape[1]],dtype=np.int8)
	mask[:] = 0

	for N in good_labels:
	    mask = mask + np.where(labels==N,1,0)
	mask = morphology.dilation(mask,np.ones([20,20])) # one last dilation
	mask1 = np.ndarray([img.shape[0],img.shape[1]],dtype=bool)
	
	mask1[mask>0]=True
	mask1[mask==0] = False
		
	return mask1

def varify_mask(nodes,masks):
	for node in nodes:
		r = np.sqrt(np.sum(node[1]**2))
		node = node[0]
		if masks[node[0],node[1],node[2]] == False:
			return False

	return True


def mask3d(f,savePath):
	
        save = False
	data = np.load(f)
	img = data[0]
	name = data[1]
	nodes = data[2]
	print nodes

        if os.path.exists('%s%s.npy'%(savePath,name)):
            print "mask exits"
            return 1 
        if save:
	    normed = norm(img)
            if not os.path.exists('./%s/'%name):
	        os.mkdir('./%s/'%name)
	    if not os.path.exists('./%s_nomask/'%name):
	        os.mkdir('./%s_nomask/'%name)

	masks=np.ndarray(shape=img.shape)
	for i in range(img.shape[2]):
		try:
		    mask = generate(img[:,:,i],1)		
		    masks[:,:,i] = mask
                except:
                    mask = np.ones(shape=masks[:,:,0].shape)
                    masks[:,:,i] = mask
                    print "%s %d mask generate False, is all True"%(name,i)
                    if save:
			masked = normed[:,:,i]
			masked[mask==0] = 0
			try:
				
#				cv2.imwrite('%s/%i.jpg'%(name,i),masked)
#				cv2.imwrite('%s_nomask/%i.jpg'%(name,i),normed[:,:,i])
				pass
			except:
				pass

        if np.sum(masks) < 0.05* np.prod(masks.shape):
            masks = np.ones(shape=masks.shape)
            print '%s masks is fault and make it all True'%name

        bmasks = np.array(shape=img.shape,dtype = bool)
        bmasks[masks==1]=True
        bmasks[masks==0]=False
        np.save('%s%s.npy'%(savePath,name),bmasks)
	
        if not varify_mask(nodes,masks):
	    print "%s mask cover a node"%name,"mask is not all right"
            return 0
        else:
            print "mask of %s is ok,saved in %s"%(name,savePath)
            return 1
        #return masks



def read_mhd(imagePath,cads,dataset,norm_flag=True,mask_flag=False,save=True):
	norm_str='normed' if norm_flag else'unnormed'
	saveDataPath = '/home/lijunying/luna17/Cache/%s_%s' % (norm_str,dataset)
	if save and not os.path.exists(saveDataPath):
	    os.mkdir(saveDataPath)
	if mask_flag:
	    saveMakPath = '/home/lijunying/luna17/Cache/mask_%s' % dataset
	    if save and not os.path.exists(saveMakPath):
			os.mkdir(saveMakPath)
        
#
	imageName = os.path.split(imagePath)[1].replace('.mhd','') 
	if os.path.exists("%s/%s.npy"%(saveDataPath,imageName)) and mask_flag==False:
	    print 'image exist in savePath'
	    return np.load("%s/%s.npy"%(saveDataPath,imageName))
	elif os.path.exists("%s/%s.npy"%(saveDataPath,imageName)) and os.path.exists("%s/%s.npy"%(saveMakPath,imageName)):
	    print 'image exist in savePath'
	    return np.load("%s/%s.npy"%(saveDataPath,imageName)),np.load("%s/%s.npy"%(saveMakPath,imageName))
	if dataset!='test' and dataset!='test2' and cads is not None:
	    indices = cads[cads['seriesuid'] == imageName].index
	    print '[',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,'](pid:',os.getpid(),')Processing ',imageName,'with', str(len(indices)) ,'patches(total images).'
	img, origin, spacing = load_itk(imagePath)
#	
#
	resize_factor = spacing / RESIZE_SPACING
	new_real_shape = img.shape * resize_factor
	new_real_shape = new_real_shape/TARGET_SPACING
	new_shape = np.round(new_real_shape)
        
	new_shape = np.array(new_shape,dtype=float)
	real_resize = new_shape / img.shape
	new_spacing = spacing / real_resize
	img = scipy.ndimage.interpolation.zoom(img, real_resize)
    
	if norm_flag:
	    img=norm(img)
	node = []
        
	if dataset!='test' and dataset!='test2' and cads is not None:
	    for i in indices:
		row = cads.iloc[i]
		label = row.diameter_mm>3
		if label > 0:
                        diameterx = float(row.diameter_mm)/target[0]
                        diameterz = float(row.diameter_mm)/target[-1]
                        diameter = np.array([diameterx,diameterx,diameterz])
                        world_coords = np.array([row.coordX, row.coordY, row.coordZ])
                        coords = world_2_voxel(world_coords,origin,new_spacing)
                        node.append([coords,diameter])
                            
	node = np.array(node,dtype=np.float)
        #print imageName,node
	savedata = np.array([img,imageName,node])

	print '[',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,'](pid:',os.getpid(),')save as',imageName,'.npy.'  
	if save:
	    np.save("%s/%s.npy"%(saveDataPath,imageName),savedata)
    
	if mask_flag: 
		if os.path.exists("%s/%s.npy"%(saveMakPath,imageName)):
			masks=np.load("%s/%s.npy"%(saveMakPath,imageName))
		else:
			masks=np.ndarray(shape=img.shape,dtype=bool)
			for i in range(img.shape[2]):
				try:
					mask = generate(img[:,:,i],1)		
					masks[:,:,i] = mask
				except:
					mask = np.ones(shape=masks[:,:,0].shape)
					masks[:,:,i] = mask
					print "%s %d mask generate False, is all True"%(imageName,i)
			np.save("%s/%s.npy"%(saveMakPath,imageName),masks)
		return savedata,masks
	else:
		return savedata

def crop(img,coords,offset=40):
	#img = np.array(img)
    coords = np.array(coords,dtype=int)
    data = np.zeros([offset*2, offset*2, offset*2],dtype=int)
    begin_img= coords - offset
    end_img= coords + offset
    begin_coord = np.maximum(begin_img,np.zeros([3],dtype=int))
    end_coord = np.minimum(end_img,np.array(img.shape))
    begin_coord_data = begin_coord - begin_img
    end_coord_data = np.array(data.shape) + end_coord - end_img 
    #print begin_coord_data,end_coord_data
    if np.max(begin_coord_data) < offset * 2 and np.min(end_coord_data) > 0:
        data[begin_coord_data[0]:end_coord_data[0],begin_coord_data[1]:end_coord_data[1],begin_coord_data[2]:end_coord_data[2]] = img[begin_coord[0]:end_coord[0],begin_coord[1]:end_coord[1],begin_coord[2]:end_coord[2]]
        return [True,data]
    else:
        return [False,[]]

if __name__ == "__main__":
    img = np.ones([6,6,6])
    c= crop(img, [5,5,5], 2)
    im=c[1]
    print im