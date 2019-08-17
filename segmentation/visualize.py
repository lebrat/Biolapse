import matplotlib.pyplot as plt
import numpy as np
import imageio
import pickle
import glob
import cv2
import os

import data_loader

"""
Display loss and accuracy during training.
"""
def display_loss(name,save=False):
	tmp=pickle.load(open(os.getcwd()+'/Data/Information/'+name+'plots.p','rb'))
	plt.figure(1)
	plt.plot(tmp['loss'][2:])
	plt.plot(tmp['val_loss'][2:])
	plt.title('model loss: '+name)
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	if save:
		if not os.path.exists(os.path.join(os.getcwd(),'Outputs')):
			os.makedirs(os.path.join(os.getcwd(),'Outputs'))
		plt.savefig(os.path.join(os.getcwd(),'Outputs',name+'_loss.png'))

	plt.figure(2)
	plt.plot(tmp['acc'][2:])
	plt.plot(tmp['val_acc'][2:])
	plt.title('model accuracy: '+name)
	plt.ylabel('acc')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	if save:
		plt.savefig(os.path.join(os.getcwd(),'Outputs',name+'_acc.png'))

	# plt.figure(3)
	# plt.plot(tmp['mean_absolute_error'][2:])
	# plt.plot(tmp['val_mean_absolute_error'][2:])
	# plt.title('model mean_absolute_error: '+name)
	# plt.ylabel('mean_absolute_error')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'test'], loc='upper left')
	# if save:
	# 	plt.savefig(os.getcwd()+'/Data/'+name+'_mae.png')

	plt.show(False)

""" 
Display temporal images with time as last channel.
"""
def display_temporal_output(model,imgs,N=2,thresh=0.):
	leng=len(imgs)
	ite=np.random.permutation(leng)
	ite=ite[:np.min([leng,N])]
	for i in ite:
		img=imgs[i]
		out = np.squeeze(model.predict(np.expand_dims(img,0)))[:,:,0]
		plt.figure(4)
		plt.imshow(np.squeeze(img[:,:,0]))
		plt.title('Original image')
		plt.figure(5)
		plt.imshow(out)
		plt.title('Prediction')
		plt.figure(6)
		plt.imshow(np.array(out>thresh,dtype=float))
		plt.title('Prediction thresholded')
		plt.show(False)

""" 
Display non temporal images with time as last channel.
"""
def display_output(model,imgs,N=2,thresh=0.):
	leng=len(imgs)
	ite=np.random.permutation(leng)
	ite=ite[:np.min([leng,N])]
	for i in ite:
		img=imgs[i]
		out = np.squeeze(model.predict(np.expand_dims(img,0)))
		plt.figure(4)
		plt.imshow(np.squeeze(img))
		plt.title('Original image')
		plt.figure(5)
		plt.imshow(out)
		plt.title('Prediction')
		plt.figure(6)
		plt.imshow(np.array(out>thresh,dtype=float))
		plt.title('Prediction thresholded')
		plt.show(False)

""" 
Display temporal images with time as last channel.
"""
def display_temporal_output_channel_first(model,imgs,N=2,thresh=0.):
	leng=len(imgs)
	ite=np.random.permutation(leng)
	ite=ite[:np.min([leng,N])]
	for i in ite:
		img=imgs[i]
		out = np.squeeze(model.predict(np.expand_dims(img,0)))[0,:,:]
		plt.figure(4)
		plt.imshow(np.squeeze(img[0,:,:]))
		plt.title('Original image')
		plt.figure(5)
		plt.imshow(out)
		plt.title('Prediction')
		plt.figure(6)
		plt.imshow(np.array(out>thresh,dtype=float))
		plt.title('Prediction thresholded')
		plt.show(False)

"""
Save images in two different names.
"""
def save_png_list(model, imgs, name='default', thresh=0, format_im=np.uint8):
	leng=len(imgs)
	t=1
	cpt=0
	pred = []
	for i in range(imgs.shape[0]):
		img=imgs[i]
		if thresh > 0:
			out = np.array(np.squeeze(model.predict(np.expand_dims(img,0)))>thresh,dtype=np.float)
		else:
			out = np.squeeze(model.predict(np.expand_dims(img,0)))
		if not os.path.exists(os.path.join('Outputs',name)):
			os.makedirs(os.path.join('Outputs',name))
		for t in range(out.shape[2]):
			tmp = np.array(np.iinfo(format_im).max*(out[:,:,t]-np.min(out[:,:,t]))/(np.max(out[:,:,t])-np.min(out[:,:,t])),dtype=format_im)
			imageio.imsave(os.path.join('Outputs',name)+'/mask_'+str(cpt).zfill(7)+'.png', tmp)
			tmp = np.array(np.iinfo(format_im).max*(img[:,:,t]-np.min(img[:,:,t]))/(np.max(img[:,:,t])-np.min(img[:,:,t])),dtype=format_im)
			imageio.imsave(os.path.join('Outputs',name)+'/original_'+str(cpt).zfill(7)+'.png', tmp)
			pred.append(out[:,:,t])
			cpt+=1
	return np.array(pred)

"""
Save images in two different names. Function for time in first channel (LSTM).
"""
def save_png_list_channel_first(model, imgs, name='default', thresh=0, format_im=np.uint8):
	leng=len(imgs)
	cpt=0
	pred = []
	for i in range(imgs.shape[0]):
		img=imgs[i]
		if thresh > 0:
			out = np.array(np.squeeze(model.predict(np.expand_dims(img,0)))>thresh,dtype=np.float)
		else:
			out = np.squeeze(model.predict(np.expand_dims(img,0)))
		if not os.path.exists(os.path.join('Outputs',name)):
			os.makedirs(os.path.join('Outputs',name))
		for t in range(out.shape[0]):
			tmp = np.array(np.iinfo(format_im).max*(out[t]-np.min(out[t]))/(np.max(out[t])-np.min(out[t])),dtype=format_im)
			imageio.imsave(os.path.join('Outputs',name)+'/mask_'+str(cpt).zfill(7)+'.png', tmp)
			tmp = np.array(np.iinfo(format_im).max*(img[t]-np.min(img[t]))/(np.max(img[t])-np.min(img[t])),dtype=format_im)
			imageio.imsave(os.path.join('Outputs',name)+'/original_'+str(cpt).zfill(7)+'.png', tmp)
			pred.append(out[t])
			cpt+=1
	return np.array(pred)

"""
Save images follow by mask to make merged video. Shoulb be apply before 'automatic_process'.
"""
def save_png_video(model, imgs, name='default', thresh=0, format_im=np.uint8,post_process=True):
	cpt=0
	for i in range(imgs.shape[0]):
		img=imgs[i]
		if thresh > 0:
			out = np.array(np.squeeze(model.predict(np.expand_dims(img,0)))>thresh,dtype=np.float)
			if post_process:
				out = np.array(elementary_post_process(out, kernel_radius=3,format_im=format_im),dtype=np.float)
		else:
			out = np.squeeze(model.predict(np.expand_dims(img,0)))
		if not os.path.exists(os.path.join('Outputs',name)):
			os.makedirs(os.path.join('Outputs',name))
		for t in range(out.shape[2]):
			tmp = np.array(np.iinfo(format_im).max*(out[:,:,t]-np.min(out[:,:,t]))/(np.max(out[:,:,t])-np.min(out[:,:,t])),dtype=format_im)
			imageio.imsave(os.path.join('Outputs',name)+'/'+str(cpt).zfill(7)+'1.png', tmp)
			tmp = np.array(np.iinfo(format_im).max*(img[:,:,t]-np.min(img[:,:,t]))/(np.max(img[:,:,t])-np.min(img[:,:,t])),dtype=format_im)
			imageio.imsave(os.path.join('Outputs',name)+'/'+str(cpt).zfill(7)+'2.png', tmp)
			cpt+=1

"""
Automatically take the output of save_png_video, remove first 'cut' frames, make 'nb_movie' movies.
The video is composed of 'nb_seq_in_video' frames. Make sure that enough frame are available.
"""
def automatic_process(name,cut=10,nb_movie=2,nb_seq_in_video=130):
	tmp = os.path.join('Outputs',name,'tmp')
	if not os.path.exists(tmp):
	    os.makedirs(tmp)
	tmp = os.path.join('Outputs',name,'tmp','test.png')
	tmp2 = os.path.join('Outputs',name,'*.png')
	os.system("montage {0} -tile 2x1 -geometry +1+1 {1}".format(tmp2,tmp))
	for im in glob.glob(os.path.join('Outputs',name,'tmp','*.png')):
		os.rename(im,os.path.join(os.path.dirname(im),os.path.basename(im).zfill(15)))
	tmp = os.path.join('Outputs',name,'tmp','*.png')
	import shutil
	for i in range(nb_movie):
		for im in glob.glob(tmp)[i*(nb_seq_in_video-cut):i*(nb_seq_in_video-cut)+cut]:
			os.remove(im)
		tmp2 = os.path.join('Outputs',name,'tmp','film{0}'.format(i))
		if not os.path.exists(tmp2):
			os.makedirs(tmp2)
		for im in glob.glob(tmp)[i*(nb_seq_in_video-cut):(i+1)*(nb_seq_in_video-cut)]:
			shutil.copy(im,os.path.join(tmp2,os.path.basename(im)))
	for i in range(nb_movie):
		tmp = os.path.join('Outputs',name,'tmp','film{0}'.format(i),'*.png')
		tmp2 = os.path.join('Outputs','film{0}_'.format(i)+name+'.gif')
		os.system("convert {0} {1}".format(tmp,tmp2))
	tmp = os.path.join('Outputs',name)
	import shutil
	shutil.rmtree(tmp, ignore_errors=True)

def elementary_post_process(im, kernel_radius=3,format_im=np.uint8):
	x = np.linspace(-1,1,2*kernel_radius)
	y = np.linspace(-1,1,2*kernel_radius)
	XX,YY = np.meshgrid(x,y)
	kernel = np.array((XX**2+YY**2) < (x[-1]+y[kernel_radius]),dtype=np.uint8)
	erosion = cv2.erode(im,kernel,iterations = 1)
	dilation = cv2.dilate(erosion,np.ones((int(2.5*kernel_radius),int(2.5*kernel_radius))),iterations = 1)

	result2 = np.logical_and(dilation,im)
	result2=np.array(result2,dtype='f')
	result2 = np.iinfo(format_im).max*(result2-np.min(result2))/(np.max(result2)-np.min(result2))
	result2=np.array(result2,dtype=format_im)

	return result2

########### Select useful ###########

# """
# Image in shape (batch_size,nx,ny,time)
# """
# def display_temporal_im(im,nb_time_serie=1):
# 	for k in range(nb_time_serie):
# 		for t in range(im.shape[3]):
# 			plt.figure()
# 			plt.imshow(im[k,:,:,t])
# 			plt.title('Figure {0}, time {1}'.format(k,t))
# 			plt.show(False)
			
# def save_mask(masks,name='default', format_im=np.uint8):
# 	if not os.path.exists(os.path.join('Labels',name)):
# 		os.makedirs(os.path.join('Labels',name))
# 	cpt = 0
# 	for i in range(masks.shape[0]):
# 		tmp = np.array(np.iinfo(format_im).max*(masks[i]-np.min(masks[i]))/(np.max(masks[i])-np.min(masks[i])),dtype=format_im)
# 		imageio.imsave(os.path.join('Labels',name)+'/'+str(cpt).zfill(7)+'.png', tmp)
# 		cpt+=1

# def save_raw_mask(masks,name='default', format_im=np.uint8):
# 	if not os.path.exists(os.path.join('Labels',name)):
# 		os.makedirs(os.path.join('Labels',name))
# 	cpt = 0
# 	for i in range(masks.shape[0]):
# 		tmp = np.array(masks[i],dtype=format_im)
# 		imageio.imsave(os.path.join('Labels',name)+'/'+str(cpt).zfill(7)+'.png', tmp)
# 		cpt+=1

# def save_mask_to_tiff(masks,name='default', format_im=np.uint8):
# 	if not os.path.exists(os.path.join('Labels',name)):
# 		os.makedirs(os.path.join('Labels',name))
# 	masks = np.array(np.iinfo(format_im).max*(masks-np.min(masks))/(np.max(masks)-np.min(masks)), dtype=format_im)
# 	imageio.mimwrite(os.path.join('Labels',name)+'/full_tiff.tif',masks)

# def save_path_track_tiff(list_data,nx=512,ny=512,format_im=np.uint8,name='default'):
# 	sigma = 3
# 	X,Y =np.meshgrid(np.linspace(-nx/2,nx/2,nx),np.linspace(-ny/2,ny/2,ny))
# 	G = np.exp(-(X**2+Y**2)/(2*sigma**2))
# 	G = np.fft.fftshift(G/np.sum(G))
# 	G = np.fft.fft2(G)
# 	max_frame=0
# 	for num in list_data:
# 		for path in num:
# 			if len(path)==1:
# 				fr = path[0]['frame']
# 			else:
# 				fr = path['frame']
# 			if fr > max_frame:
# 				max_frame = fr
# 	out_path = np.zeros((max_frame-1,nx,ny))
# 	for num in list_data:
# 		for path in num:
# 			if len(path)==1:
# 				p = path[0]
# 			else:
# 				p = path
# 			out_path[p['frame']-1,int(p['x']),int(p['y'])] = 1

# 	for i in range(max_frame-1):
# 		out_path[i] = np.real(np.fft.ifft2(G*np.fft.fft2(out_path[i]))) 
# 		out_path[i] = np.array(np.iinfo(format_im).max*(out_path[i]-np.min(out_path[i]))/(np.max(out_path[i])-np.min(out_path[i])), dtype=format_im)

# 	out_path = np.array(out_path, dtype=format_im)

# 	if not os.path.exists(os.path.join('Labels',name)):
# 		os.makedirs(os.path.join('Labels',name))
# 	imageio.mimwrite(os.path.join('Labels',name)+'/path_tiff.tif',out_path)


# """
# Post process temporal data.
# Remove small artefacts.
# Erode and dilate images, then a logical and is done with the true mask. Small areas should have been deleted in the erode-dilate process.
# Erode: remove pixels which haven't all pixels in the neighbourhood equal to 1.
# Dilate: expand image of same size that it has been erode. 

# If 'compute_imgs' is False, the following parameters wont be used.
# """
# def post_process_temporal(path_imgs, kernel_radius=3, start_code='mask_', compute_imgs=False, model='', name='default', format_im=np.uint8, nx=256, ny=256, TIME=10, thresh=0.5):
# 	if compute_imgs or model=='':
# 		for (fold, subfold, files) in os.walk(path_imgs):
# 			if os.path.basename(fold)!='PostProcess' and os.path.basename(fold)!='Tracking':
# 				masks = [f for f in files if f.endswith('.png') and f.startswith(start_code)]
# 				for m in masks:
# 					im = np.array(imageio.imread(os.path.join(fold,m))>np.iinfo(format_im).max/2, dtype=np.float)

# 					x = np.linspace(-1,1,2*kernel_radius)
# 					y = np.linspace(-1,1,2*kernel_radius)
# 					XX,YY = np.meshgrid(x,y)
# 					kernel = np.array((XX**2+YY**2) < (x[-1]+y[kernel_radius]),dtype=np.uint8)
# 					erosion = cv2.erode(im,kernel,iterations = 1)
# 					dilation = cv2.dilate(erosion,np.ones((int(2.5*kernel_radius),int(2.5*kernel_radius))),iterations = 1)

# 					result2 = np.logical_and(dilation,im)
# 					result2=np.array(result2,dtype='f')
# 					result2 = np.iinfo(format_im).max*(result2-np.min(result2))/(np.max(result2)-np.min(result2))
# 					result2=np.array(result2,dtype=format_im)

# 					if not os.path.exists(os.path.join(fold,'PostProcess')):
# 						os.makedirs(os.path.join(fold,'PostProcess'))
# 					imageio.imsave(os.path.join(fold,'PostProcess',m),result2)

# 	else:
# 		X = data_loader.generator_temps_no_mask(path_imgs,nx=nx,ny=ny,TIME = TIME, format_im=format_im)
# 		X = np.expand_dims(np.array(X,dtype=np.float)/np.iinfo(format_im).max,4)
# 		save_png_list(model, X, name=name, thresh=0, format_im=np.uint8)

# 		for (fold, subfold, files) in os.walk(path_imgs):
# 			if os.path.basename(fold)!='PostProcess' and os.path.basename(fold)!='Tracking':
# 				masks = [f for f in files if f.endswith('.png') and f.startswith('mask_')]
# 				for m in masks:
# 					im = np.array(imageio.imread(os.path.join(fold,m))>np.iinfo(format_im).max/2, dtype=np.float)

# 					x = np.linspace(-1,1,2*kernel_radius)
# 					y = np.linspace(-1,1,2*kernel_radius)
# 					XX,YY = np.meshgrid(x,y)
# 					kernel = np.array((XX**2+YY**2) < (x[-1]+y[kernel_radius]),dtype=np.uint8)
# 					erosion = cv2.erode(im,kernel,iterations = 1)
# 					dilation = cv2.dilate(erosion,np.ones((2*kernel_radius,2*kernel_radius)),iterations = 1)

# 					result2 = np.logical_and(dilation,im)
# 					result2=np.array(result2,dtype='f')
# 					result2 = np.iinfo(format_im).max*(result2-np.min(result2))/(np.max(result2)-np.min(result2))
# 					result2=np.array(result2,dtype=format_im)

# 					if not os.path.exists(os.path.join(fold,'PostProcess')):
# 						os.makedirs(os.path.join(fold,'PostProcess'))
# 					imageio.imsave(os.path.join(fold,'PostProcess',m),result2)

# def mask_to_label(path_imgs):
# 	for (fold, subfold, files) in os.walk(path_imgs):
# 		if os.path.basename(fold)!='Labels':
# 			masks = sorted([f for f in files if f.endswith('.png') and f.startswith(start_code)], key = stringSplitByNumbers)
# 			structure = np.ones((3, 3), dtype=np.int)
# 			for m in masks:
# 				if not os.path.exists(os.path.join(fold,'Labels')):
# 					os.makedirs(os.path.join(fold,'Labels'))
# 				im = np.array(imageio.imread(os.path.join(fold,m))>np.iinfo(format_im).max/2, dtype=np.float)
# 				map_class, ncomponents = label(im, structure)
# 				imageio.imsave(os.path.join(fold,'Labels',m),map_class)

# """
# Tracking function. Attribute one number to each connected component.
# A merge is validate if cells have spend 'merge_time' together.
# """
# def tracking(path_imgs):
# 	map_final = list_data = 0
# 	for (fold, subfold, files) in os.walk(path_imgs):
# 		if os.path.basename(fold)!='Tracking':
# 			masks = sorted([f for f in files if f.endswith('.png') and f.startswith('mask_')], key = stringSplitByNumbers)
# 			structure = np.ones((3, 3), dtype=np.int)
# 			structure = np.array([[0,1,0], [1,1,1],[0,1,0]])
# 			if len(masks)>0:
# 				im = imageio.imread(os.path.join(fold,masks[0]))
# 				X = np.zeros((len(masks),im.shape[0],im.shape[1])) 
# 				cpt = 0
# 				for m in masks:
# 					X[cpt] = np.array(imageio.imread(os.path.join(fold,m))>np.iinfo(format_im).max/2, dtype=np.float)
# 					cpt+=1

# 				map_final = np.zeros_like(X)
# 				list_data = []
# 				for i in range(len(masks)):
# 					if i==0:
# 						map_class, ncomponents = label(X[i], structure)
# 						map_final[i] = map_class
# 						for j in range(1,ncomponents):
# 							mask_tmp=np.array(map_class==j,dtype='f')
# 							x,y = scipy.ndimage.measurements.center_of_mass(mask_tmp)
# 							feat_dict = {'id':j,
# 							 'x': x,
# 							 'y': y,
# 							 'pixel_volume': np.sum(mask_tmp>0),
# 							 'frame': i
# 							}
# 							list_data.append([feat_dict])
# 						last_free_idx = ncomponents-1


# 					else:
# 						map_class, ncomponents = label(X[i], structure)
# 						displ1 = (X[i-1]-X[i])>0
# 						displ2 = (X[i-1]-X[i])<0
# 						# displ3 = (X[i+1]-X[i])>0
# 						# displ4 = (X[i+1]-X[i])<0

# 						kernel_radius=9
# 						thresh1=250
# 						thresh2=40
# 						x = np.linspace(-1,1,2*kernel_radius)
# 						y = np.linspace(-1,1,2*kernel_radius)
# 						XX,YY = np.meshgrid(x,y)
# 						kernel = np.array((XX**2+YY**2) < (x[-1]+y[kernel_radius]),dtype=np.uint8)
# 						erosion = cv2.erode(im,kernel,iterations = 1)
# 						dilation = cv2.dilate(erosion,np.ones((int(2.5*kernel_radius),int(2.5*kernel_radius))),iterations = 1)

# 						result1 = np.logical_and(dilation,displ1)
# 						result2 = np.logical_and(dilation,displ2)
# 						# result1 = np.logical_and(dilation,displ3)
# 						# result2 = np.logical_and(dilation,displ4)
# 						result1=np.array(result1,dtype='f')
# 						result2=np.array(result2,dtype='f')
# 						# result3=np.array(result3,dtype='f')
# 						# result4=np.array(result4,dtype='f')

# 						for k in range(1):
# 								map_class_tmp1, ncomponents_tmp1 = label(result1, structure)
# 								map_class_tmp2, ncomponents_tmp2 = label(result2, structure)
# 								# map_class_tmp3, ncomponents_tmp3 = label(result3, structure)
# 								# map_class_tmp4, ncomponents_tmp4 = label(result4, structure)
# 								for j in range(1,ncomponents_tmp1):
# 									if np.sum(map_class_tmp1==j)<=thresh1:
# 										result1[map_class_tmp1==j]=0
# 								for j in range(1,ncomponents_tmp2):
# 									if np.sum(map_class_tmp2==j)<=thresh1:
# 										result2[map_class_tmp2==j]=0
# 								# for j in range(1,ncomponents_tmp3):
# 								# 	if np.sum(map_class_tmp3==j)<=thresh1:
# 								# 		result3[map_class_tmp3==j]=0
# 								# for j in range(1,ncomponents_tmp4):
# 								# 	if np.sum(map_class_tmp4==j)<=thresh1:
# 								# 		result4[map_class_tmp4==j]=0

# 						idx = list(np.arange(1,last_free_idx+1))
# 						# Disparition
# 						# if np.sum(result1)>thresh1:



# 						# 	tmp = map_final[i-1]
# 						# 	idx_to_remove = int(np.mean(tmp[result1>0]))
# 						# 	feat_dict = {'id':idx_to_remove,
# 						# 	 'x': 0,
# 						# 	 'y': 0,
# 						# 	 'pixel_volume': 0,
# 						# 	 'frame': i
# 						# 	}
# 						# 	list_data[idx_to_remove-1].append([feat_dict])

# 						# # Apparition
# 						# if np.sum(result2)>thresh1:
# 						# 	idx_to_add = int(np.mean(map_class[result2>0]))
# 						# 	mask_tmp = map_class
# 						# 	mask_tmp[map_class!=idx_to_add] = 0
# 						# 	x,y = scipy.ndimage.measurements.center_of_mass(mask_tmp)
# 						# 	feat_dict = {'id':last_free_idx,
# 						# 	 'x': x,
# 						# 	 'y': y,
# 						# 	 'pixel_volume': np.sum(mask_tmp>0),
# 						# 	 'frame': i
# 						# 	}
# 						# 	list_data.append([feat_dict])
# 						# 	last_free_idx += 1

# 						# Merge detection
# 						for j in range(1,ncomponents+1):
# 							mask_tmp=np.array(map_class==j,dtype='f')
# 							if np.sum(mask_tmp)>thresh2: # Si trop petit on le regarde meme pas
# 								dilation = cv2.dilate(map_final[i-1],np.ones((int(1*kernel_radius),int(1*kernel_radius))),iterations = 1)
# 								result = mask_tmp*dilation
# 								if np.sum(result>0)==0: # Ajout si on le retrouve pas avant
# 									x,y = scipy.ndimage.measurements.center_of_mass(mask_tmp)
# 									feat_dict = {'id':last_free_idx,
# 									 'x': x,
# 									 'y': y,
# 									 'pixel_volume': np.sum(mask_tmp>0),
# 									 'frame': i
# 									}
# 									list_data.append([feat_dict])
# 									map_final[i]+=mask_tmp*last_free_idx
# 									last_free_idx += 1
# 								else:
# 									idx_previous_frame = int(np.median(result[result>0]))
# 									if idx_previous_frame in idx:
# 										x,y = scipy.ndimage.measurements.center_of_mass(mask_tmp)
# 										feat_dict = {'id':idx_previous_frame,
# 										 'x': x,
# 										 'y': y,
# 										 'pixel_volume': np.sum(mask_tmp>0),
# 										 'frame': i
# 										}
# 										map_final[i]+=mask_tmp*idx_previous_frame
# 										# print(idx_previous_frame,len(list_data))
# 										list_data[idx_previous_frame-1].append([feat_dict])
# 										idx.remove(idx_previous_frame)
# 									else: #Si son indice a deja ete utilisé mais que le merge a cassé on en refait un autre. TODO: retrouver l'indice avec les frames precedentes.
# 										x,y = scipy.ndimage.measurements.center_of_mass(mask_tmp)
# 										feat_dict = {'id':last_free_idx,
# 										 'x': x,
# 										 'y': y,
# 										 'pixel_volume': np.sum(mask_tmp>0),
# 										 'frame': i
# 										}
# 										list_data.append([feat_dict])
# 										map_final[i]+=mask_tmp*last_free_idx
# 										last_free_idx += 1
# 	return map_final, list_data





# from skimage.transform import resize
# path_imgs ='/media/valentin/Data/Documents/Projets/ML/mldenoise/Keras/Labels/default'
# """
#   - path_true: path where to find the true data as 'original_{number}.png'
#   - 

# TODO:
#   -improve tracking, be able to retrieve cell after merge.
# """
# def crop_track(map_final,path_true,name='default_tiff',format_im=np.uint8):
# 	path_true = '/media/valentin/Data/Documents/Projets/ML/mldenoise/Keras/Outputs/test_track'
# 	max_idx=np.max(map_final)
# 	num = 35
# 	st = True
# 	nx = ny = 64
# 	delta = 15
# 	li = []
# 	ma = []
# 	for i in range(map_final.shape[0]):
# 		crt_mask = np.array(map_final[i]==num,dtype='f')
# 		if np.sum(crt_mask)==0:
# 			if not(st):
# 				break
# 		else:
# 			im = np.array(imageio.imread(path_true+'/original_'+str(i).zfill(7)+'.png'), dtype=np.float)
# 			idx = np.where(crt_mask==1) 
# 			tmp_img = im[np.min(idx[0])-delta:np.max(idx[0]+delta),np.min(idx[1])-delta:np.max(idx[1])+delta]
# 			li.append(resize(tmp_img,(nx,ny)))
# 			ma.append(resize(crt_mask[np.min(idx[0])-delta:np.max(idx[0]+delta),np.min(idx[1])-delta:np.max(idx[1])+delta],(nx,ny)))
	
# 	li = np.array(li)
# 	ma = np.array(ma)
# 	save_mask_to_tiff(li,name='default', format_im=np.uint16)
# 	save_mask_to_tiff(ma,name='default2', format_im=np.uint8)



# path_imgs = '/media/valentin/Data/Documents/Projets/ML/mldenoise/Keras/Labels/default/full_tiff.tif'
# path_mask = '/media/valentin/Data/Documents/Projets/ML/mldenoise/Keras/Labels/default2/full_tiff.tif'
# """
# TODO: do the same with only the interior of the cell
# """
# def tiff_to_hist(path_imgs, path_mask):
# 		im = np.array(imageio.mimread(path_imgs),dtype='f')
# 		ma = np.array(imageio.mimread(path_mask),dtype='f')
# 		for i in range(im.shape[0]):
# 			im_crt = im[i]/np.max(im[i])
# 			ma_crt = ma[i]
# 			im_crt = im_crt[ma_crt>0]
# 			# im_crt = resize(im_crt,(im_crt.shape[0]*im_crt.shape[1],1))
# 			plt.hist(im_crt,normed =1 ,bins = 100, range = (0,1), label ="test",color='w',edgecolor='r')
# 			plt.savefig('test/'+'hist'+str(i)+'.png')
# 			plt.close('all')




# def main():

# 	nx = ny = 256
# 	TIME = 10
# 	format_im = np.uint16
# 	thresh=0.1
# 	path_imgs = '/media/valentin/Data/Documents/These/Data/tracking/Thomas/16bits'

# 	# name='NN_Unet3D_filt1_depth4_Thomas'
# 	# display_loss(name)
# 	# name='NN_Unet3D_filt2_depth7_Thomas'
# 	# display_loss(name)
# 	# name='NN_Unet3D_filt4_depth2_Thomas'
# 	# display_loss(name)
# 	# name='NN_Unet3D_filt4_depth4_Thomas'
# 	# display_loss(name)
# 	# name='NN_Unet3D_filt16_depth4_Thomas'
# 	# display_loss(name)

# 	name='test_track'
# 	path_h5 = 'Data/Model_h5/NN_Unet3D_filt4_depth4_Thomas.h5'
# 	X = generator_temps_no_mask(path_imgs,nx=nx,ny=ny,TIME = TIME, format_im=format_im)
# 	X = np.expand_dims(np.array(X,dtype=np.float)/np.iinfo(format_im).max,4)
# 	model = load_model(path_h5)
# 	save_png_list(model, X, thresh=0.2, format_im=format_im,name=name)
# 	automatic_process(name)

# 	p = '/media/valentin/Data/Documents/Projets/ML/mldenoise/Keras/Outputs/masks'
# 	map_final, list_data = tracking(p)
# 	save_raw_mask(map_,name='default', format_im=format_im)
