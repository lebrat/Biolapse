from PIL import Image
from skimage.transform import resize
from skimage import transform
from skimage.io import imread
import os
import sys
import glob
import numpy as np
import imageio
import keras

import utils

"""
Implemented:
	- Generator 
		- temporal data.
	- utils
		- string ordering
		- save array to npy

TODO: 
	- generator for classical image (no temporal).
	- augmentation data from png folder.

"""

# """
# Load images with extensions 'format_im' provided by 'path' and return array with type 'type_im'.
# 'format_im' should be readable by imageio.imread (no tiff).

# Example:
# # Load one image.
# img_to_array('im.png')
# # Load several images.
# img_to_array('path/*.png')
# """
# def img_to_array(path,type_im=np.uint8,format_im='png'):
#     imgs = glob.glob(path)
#     im = imageio.imread(imgs[0])
#     im = np.zeros((im.shape[0],im.shape[1],len(imgs)))
#     cpt = 0
#     for i in glob.glob(path):
#     	if imgs[i].endswith(format_im):
#         	im[:,:,i] = utils.rgb_to_gray(np.array(imageio.imread(imgs[i]),dtype=type_im))
#         	cpt += 1
#     im = im[:,:,:cpt+1]
#     return im


# def resize_im(img, shape):
# 	return resize(img, shape, mode='constant', preserve_range=True)

# def normalize_im(img):
# 	return (img-img.min())/(img.max()-img.min())


# """
# Load images with extensions tiff or tif provided by 'path' and return array with type 'type_im'.

# Example:
# # Load one image.
# tiff_to_array('im.tiff')
# # Load several images.
# tiff_to_array('path/*.tiff')
# """
# def tiff_to_array(path,type_im=np.uint8):
#     imgs = glob.glob(path)
#     im = np.moveaxis(utils.rgb_to_gray(np.array(imageio.mimread(imgs[0],memtest=False),dtype=type_im)),0,-1)
#     im = np.zeros((im.shape[0],im.shape[1],im.shape[2],len(imgs)))
#     cpt = 0
#     for i in glob.glob(path):
#     	if imgs[i].endswith('tiff') or imgs[i].endswith('tif'):
#         	im[:,:,:,i] = np.moveaxis(utils.rgb_to_gray(np.array(imageio.mimread(imgs[i],memtest=False),dtype=type_im)),0,-1)
#         	cpt += 1
#     im = im[:,:,:,:cpt+1]
#     if im.shape[3]==1:
#         im = np.squeeze(im,3)
#     return im


# """
# Save tiff images of or set of images into png file. Parameters such as format can be specify.
# 'path_tiff' is either a tiff file or a folder containing tiff files.
# """
# def sequence_to_png(path_tiff,path_png,nx=256,ny=256,type_im=np.uint8,sample_rate=1,t_end_=1e8):
# 	if path_tiff[-3:]=='.tif' or path_tiff[-4:]=='.tiff':
# 		im = np.array(imageio.mimread(path_tiff,memtest=False),dtype=type_im)
# 		t_end = int(np.min((t_end_,im.shape[0])))
# 		im = transform.resize(np.squeeze(im),(t_end,nx,ny))
# 		im = im[:t_end]

# 		im_resize = np.zeros(im.shape,dtype=type_im)
# 		if path_png[-3:]=='tif' or path_png[-4:]=='tiff':
# 			if not os.path.exists(path_png[:path_png.rfind('/')]):
# 				os.makedirs(path_png[:path_png.rfind('/')])
# 			for t in range(0,im.shape[0],sample_rate):
# 				im_resize[t] = x*(im[t]-np.min(im[t])+1e-15)/(np.max(im[t])-np.min(im[t])+1e-15)
# 				imageio.imsave(path_png[:path_png.rfind('/')]+'/'+path_png[path_png.rfind('/')+1:][:path_png[path_png.rfind('/')+1:].rfind('.')]+str(t).zfill(7)+'.png',np.array(im_resize[t],dtype=type_im))
# 		else:
# 			if not os.path.exists(path_png):
# 				os.makedirs(path_png)
# 			for t in range(0,im.shape[0],sample_rate):
# 				im_resize[t] = np.iinfo(type_im).max*(im[t]-np.min(im[t])+1e-15)/(np.max(im[t])-np.min(im[t])+1e-15)
# 				imageio.imsave(os.path.join(path_png,path_tiff[path_tiff.rfind('/')+1:],str(t).zfill(7)+'.png'),np.array(im_resize[t],dtype=type_im))
# 	else:
# 		cpt = 0
# 		for (fold, subfold, files) in os.walk(path_tiff):
# 			for f in sorted(files,key=utils.stringSplitByNumbers):
# 				path_save = os.path.join(path_png,fold[fold.rfind('/')+1:])
# 				if not os.path.exists(path_save):
# 					os.makedirs(path_save)
# 				if f.endswith('.tif') or f.endswith('.tiff'):
# 					im = np.array(imageio.mimread(os.path.join(fold,f),memtest=False),dtype=float)
# 					t_end = int(np.min((t_end_,im.shape[0])))
# 					if len(im.shape)!=4:
# 						im = transform.resize(im,(t_end,nx,ny))
# 						im = im[:t_end]
# 						im_resize = np.zeros(im.shape,dtype=float)
# 						for t in range(0,im.shape[0],sample_rate):
# 							im_resize[t] = np.iinfo(type_im).max*(im[t]-np.min(im[t])+1e-15)/(np.max(im[t])-np.min(im[t])+1e-15)
# 							imageio.imsave(os.path.join(path_save,str(cpt).zfill(7)+'.png'),np.array(np.squeeze(im_resize[t]),dtype=type_im))
# 							cpt += 1


# def tiff_to_png_multiple_axis(path_tiff,path_save,nx=256,nb_z=1,nb_time=1,type_im=np.uint8):
# 	im = np.squeeze(np.array(imageio.mimread(path_tiff,memtest=False),dtype=float))
# 	path_save = os.path.join(path_save,path_tiff[path_tiff.rfind('/')+1:])
# 	cpt = 0
# 	for t in range(nb_time):
# 		for z in range(nb_z):
# 			if not os.path.exists(os.path.join(path_save,str(z))):
# 				os.makedirs(os.path.join(path_save,str(z)))
# 			tmp = np.iinfo(type_im).max*(im[cpt]-np.min(im[cpt])+1e-15)/(np.max(im[cpt])-np.min(im[cpt])+1e-15)
# 			tmp = transform.resize(tmp,(nx,nx))
# 			imageio.imsave(os.path.join(path_save,str(z),str(t).zfill(7)+'.png'),np.array(tmp,dtype=type_im))
# 			cpt += 1


# """
# Run through path and aggregate time images into batch of specify size.
# """
# def path_to_time_batch(path_fold,nx=128,ny=128,TIME=15,type_im=np.uint8,format_im='png'):
# 	cpt_serie = 0
# 	for (fold, subfold, files) in os.walk(path_fold):
# 		nb_serie = int(len([f for f in files if f.endswith(format_im)])/TIME)
# 		cpt_serie += nb_serie

# 	X_train = np.zeros((cpt_serie, nx, ny, TIME), dtype=type_im)
# 	Y_train = np.zeros((cpt_serie, nx, ny, TIME), dtype=type_im)
# 	cpt_serie = 0
# 	for (fold, subfold, files) in os.walk(path_fold):
# 		nb_serie = int(len([f for f in files if f.endswith(format_im)])/TIME)
# 		images_path = sorted([os.path.join(fold,f) for f in files if f.endswith('.png')], key = utils.stringSplitByNumbers)
# 		cpt =0
# 		for i in range(nb_serie):
# 			for t in range(TIME):
# 				img = imread(images_path[cpt])
# 				img = resize(img, (nx, ny), mode='constant', preserve_range=True)
# 				X_train[cpt_serie,:,:,t]=img
# 				cpt += 1
# 			cpt_serie +=1
# 	print('Data set done!')
# 	return X_train

# def rename_fold(path_fold):
# 	cpt = 0
# 	for (fold, subfold, files) in os.walk(path_fold):
# 		for f in sorted(files,key=utils.stringSplitByNumbers):
# 			if f.endswith('.png') :
# 				os.rename(os.path.join(fold,f),os.path.join(fold,str(cpt).zfill(8)+'.png'))
# 				cpt += 1


"""
Convert tif images into png images.

Example:
	path_='./'
	path_to_save='./Data'
	type_im = np.uint16
	nx = ny = 256
	tif2png(path,path_to_save,type_im,nx,ny)
"""


def tif2png(path_, path_to_save, type_im, nx, ny):
    # folders = [o for o in glob.glob(os.path.join(path_,'*')) if os.path.isdir(o)]
    for dirpath, dirnames, filenames in os.walk(path_):
        for filename in [f for f in filenames if f.endswith(".tif")]:
            im = np.array(imageio.mimread(os.path.join(
                dirpath, filename), memtest=False), dtype=type_im)
            im_resize = np.zeros((im.shape[1], nx, ny), dtype=type_im)
            for t in range(im.shape[1]):
                im_resize[t] = resize(im[0, t], (nx, ny),
                                      mode='constant', preserve_range=True)
                im_resize[t] = np.array(np.iinfo(type_im).max*(im_resize[t]-np.min(im_resize[t])+1e-15)/(
                    np.max(im_resize[t])-np.min(im_resize[t])+1e-15), dtype=type_im)
            name_save_img = os.path.join(path_to_save, filename[:-4])
            if not os.path.exists(name_save_img):
                os.makedirs(name_save_img)
            for t in range(im.shape[1]):
                imageio.imsave(os.path.join(
                    name_save_img, str(t)+'.png'), im_resize[t])


"""
Run through path and aggregate time images into batch of specify size.
"""


def path_to_time_batchs(path_fold, nx=128, ny=128, TIME=15, type_im=np.uint8, format_im='png', code_im1='images', code_im2='masks'):
    cpt_serie = 0
    for (fold, subfold, files) in os.walk(path_fold):
        if os.path.basename(fold) == code_im1:
            nb_serie = int(
                len([f for f in files if f.endswith(format_im)])/TIME)
            cpt_serie += nb_serie

    X_train = np.zeros((cpt_serie, nx, ny, TIME), dtype=type_im)
    Y_train = np.zeros((cpt_serie, nx, ny, TIME), dtype=type_im)
    cpt_serie = 0
    for (fold, subfold, files) in os.walk(path_fold):
        if os.path.basename(fold) == code_im1:
            nb_serie = int(
                len([f for f in files if f.endswith(format_im)])/TIME)
            images_path = sorted([os.path.join(fold, f) for f in files if f.endswith(
                format_im)], key=utils.stringSplitByNumbers)
            cpt = 0
            for i in range(nb_serie):
                for t in range(TIME):
                    img = imageio.imread(images_path[cpt])
                    img = resize(img, (nx, ny), mode='constant',
                                 preserve_range=True)
                    X_train[cpt_serie, :, :, t] = img
                    mask = imageio.imread(
                        images_path[cpt].replace(code_im1, code_im2))
                    mask = resize(mask, (nx, ny), mode='constant',
                                  preserve_range=True)
                    mask = np.array(np.iinfo(type_im).max *
                                    (mask > 0.01), dtype=type_im)
                    Y_train[cpt_serie, :, :, t] = mask
                    cpt += 1
                cpt_serie += 1
    print('Data set done!')
    return X_train, Y_train


"""
Run through path and return batch images.
"""


def path_to_batchs(path_fold, nx=128, ny=128, type_im=np.uint8, format_im='png', code_im1='images', code_im2='masks'):
    cpt_serie = 0
    for (fold, subfold, files) in os.walk(path_fold):
        if os.path.basename(fold) == code_im1:
            cpt_serie += int(len([f for f in files if f.endswith(format_im)]))

    X_train = np.zeros((cpt_serie, nx, ny, 1), dtype=type_im)
    Y_train = np.zeros((cpt_serie, nx, ny, 1), dtype=type_im)
    cpt_serie = 0
    for (fold, subfold, files) in os.walk(path_fold):
        if os.path.basename(fold) == code_im1:
            nb_im = int(len([f for f in files if f.endswith(format_im)]))
            images_path = sorted([os.path.join(fold, f) for f in files if f.endswith(
                format_im)], key=utils.stringSplitByNumbers)
            cpt = 0
            for i in range(nb_im):
                img = imageio.imread(images_path[cpt])
                img = resize(img, (nx, ny), mode='constant',
                             preserve_range=True)
                X_train[cpt_serie, :, :, 0] = img
                mask = imageio.imread(
                    images_path[cpt].replace(code_im1, code_im2))
                mask = resize(mask, (nx, ny), mode='constant',
                              preserve_range=True)
                mask = np.array(np.iinfo(type_im).max *
                                (mask > 0.01), dtype=type_im)
                Y_train[cpt_serie, :, :, 0] = mask
                cpt += 1
                cpt_serie += 1
    print('Data set done!')
    return X_train, Y_train


"""
Rewrite png images into npy array for faster (??) processing.
"""


def array_to_npy(x, y, test=False, name='default'):
    if not os.path.exists(os.path.join(os.getcwd(), 'tmp')):
        os.makedirs(os.path.join(os.getcwd(), 'tmp'))
    if not os.path.exists(os.path.join(os.getcwd(), 'tmp', name, 'images')):
        os.makedirs(os.path.join(os.getcwd(), 'tmp', name, 'images'))
    if not os.path.exists(os.path.join(os.getcwd(), 'tmp', name, 'masks')):
        os.makedirs(os.path.join(os.getcwd(), 'tmp', name, 'masks'))
    # for f in glob.glob(os.path.join(os.getcwd(),'tmp',name,'*')):
    #     os.shutil.rmtree()(f)
    List_id = []
    for c in range(x.shape[0]):
        if test:
            np.save(os.path.join(os.getcwd(), 'tmp', name,
                                 'images', str(99999999+c)+'.npy'), x[c])
            np.save(os.path.join(os.getcwd(), 'tmp', name,
                                 'masks', str(99999999+c)+'.npy'), y[c])
            List_id.append(99999999+c)
        else:
            np.save(os.path.join(os.getcwd(), 'tmp',
                                 name, 'images', str(c)+'.npy'), x[c])
            np.save(os.path.join(os.getcwd(), 'tmp',
                                 name, 'masks', str(c)+'.npy'), y[c])
            List_id.append(c)
    return List_id


"""
Data generator for temporal data. Takes list of id where to find the npy data.
"""


class DataGenerator3D(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size=32, dim=(256, 256, 15), n_channels=1,
                 shuffle=True, name='default'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = 1
        self.shuffle = shuffle
        self.on_epoch_end()
        self.name = name

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = np.zeros((self.batch_size, *self.dim))
        Y = np.zeros((self.batch_size, *self.dim))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i] = np.load(os.getcwd()+'/tmp/'+self.name +
                           '/images/' + str(ID) + '.npy')
            # Store class
            Y[i] = np.load(os.getcwd()+'/tmp/'+self.name +
                           '/masks/' + str(ID) + '.npy')
        return X, Y


# Augmentation
'''
Process augmentation on data.
 - 'path_imgs' is th location of true images.
 - 'path_masks' is the location of all the associated masks.
 - 'path_save' is the path to save augmentation.

Implement: 
 - 'original': no processing
 - 'oneOfTwo': keep one image over two.
 - 'reverse': read sequence in reversed direction.
 - 'reverseOneOfTwo': read one image over two in reversed direction.
 - 'rotation1', 'rotation2', 'rotation3': angle of rotation.
 - 'rotation4', 'rotation5': angle of rotation with one frame over two. 
 - 'rotation6', 'rotation7': ongle of rotation with image is reversed order.
'''


def augmentation_png(path_imgs, path_masks, path_save, original=True, oneOfTwo=True, reverse=True, reverseOneOfTwo=True,
                     rotation1=30, rotation2=150, rotation3=240, rotation4=90, rotation5=290, rotation6=120, rotation7=330):

    true_imgs = sorted(glob.glob(path_imgs), key=utils.stringSplitByNumbers)
    true_masks = sorted(glob.glob(path_masks), key=utils.stringSplitByNumbers)
    assert(len(true_masks) == len(true_imgs))
    k_max = len(true_imgs)
    if not os.path.exists(path_save):
        os.makedirs(path_save)

    if original:
        if not os.path.exists(os.path.join(path_save, 'Original')):
            os.makedirs(os.path.join(path_save, 'Original'))
        if not os.path.exists(os.path.join(path_save, 'Original', 'masks')):
            os.makedirs(os.path.join(path_save, 'Original', 'masks'))
        if not os.path.exists(os.path.join(path_save, 'Original', 'images')):
            os.makedirs(os.path.join(path_save, 'Original', 'images'))
        for k in range(len(true_imgs)):
            im = imageio.imread(true_masks[k])
            imageio.imsave(os.path.join(path_save, 'Original',
                                        'masks', str(k).zfill(10)+'.png'), im)
            im_true = imageio.imread(true_imgs[k])
            imageio.imsave(os.path.join(path_save, 'Original',
                                        'images', str(k).zfill(10)+'.png'), im_true)

    if oneOfTwo:
        if not os.path.exists(os.path.join(path_save, 'OneOfTwo')):
            os.makedirs(os.path.join(path_save, 'OneOfTwo'))
        if not os.path.exists(os.path.join(path_save, 'OneOfTwo', 'masks')):
            os.makedirs(os.path.join(path_save, 'OneOfTwo', 'masks'))
        if not os.path.exists(os.path.join(path_save, 'OneOfTwo', 'images')):
            os.makedirs(os.path.join(path_save, 'OneOfTwo', 'images'))
        for k in range(len(true_imgs)):
            if k % 2 == 0:
                im = imageio.imread(true_masks[k])
                imageio.imsave(os.path.join(path_save, 'OneOfTwo',
                                            'masks', str(k).zfill(10)+'.png'), im)
                im_true = imageio.imread(true_imgs[k])
                imageio.imsave(os.path.join(path_save, 'OneOfTwo',
                                            'images', str(k).zfill(10)+'.png'), im_true)

    if reverse:
        if not os.path.exists(os.path.join(path_save, 'Reverse')):
            os.makedirs(os.path.join(path_save, 'Reverse'))
        if not os.path.exists(os.path.join(path_save, 'Reverse', 'masks')):
            os.makedirs(os.path.join(path_save, 'Reverse', 'masks'))
        if not os.path.exists(os.path.join(path_save, 'Reverse', 'images')):
            os.makedirs(os.path.join(path_save, 'Reverse', 'images'))
        for k in range(len(true_imgs)):
            im = imageio.imread(true_masks[k_max-1-k])
            imageio.imsave(os.path.join(path_save, 'Reverse',
                                        'masks', str(k).zfill(10)+'.png'), im)
            im_true = imageio.imread(true_imgs[k_max-1-k])
            imageio.imsave(os.path.join(path_save, 'Reverse',
                                        'images', str(k).zfill(10)+'.png'), im_true)

    if reverseOneOfTwo:
        if not os.path.exists(os.path.join(path_save, 'ReverseOneOfTwo')):
            os.makedirs(os.path.join(path_save, 'ReverseOneOfTwo'))
        if not os.path.exists(os.path.join(path_save, 'ReverseOneOfTwo', 'masks')):
            os.makedirs(os.path.join(path_save, 'ReverseOneOfTwo', 'masks'))
        if not os.path.exists(os.path.join(path_save, 'ReverseOneOfTwo', 'images')):
            os.makedirs(os.path.join(path_save, 'ReverseOneOfTwo', 'images'))
        for k in range(len(true_imgs)):
            if k % 2 == 0:
                im = imageio.imread(true_masks[k_max-1-k])
                imageio.imsave(os.path.join(
                    path_save, 'ReverseOneOfTwo', 'masks', str(k).zfill(10)+'.png'), im)
                im_true = imageio.imread(true_imgs[k_max-1-k])
                imageio.imsave(os.path.join(
                    path_save, 'ReverseOneOfTwo', 'images', str(k).zfill(10)+'.png'), im_true)

    if rotation1:
        if not os.path.exists(os.path.join(path_save, 'Rotation1')):
            os.makedirs(os.path.join(path_save, 'Rotation1'))
        if not os.path.exists(os.path.join(path_save, 'Rotation1', 'masks')):
            os.makedirs(os.path.join(path_save, 'Rotation1', 'masks'))
        if not os.path.exists(os.path.join(path_save, 'Rotation1', 'images')):
            os.makedirs(os.path.join(path_save, 'Rotation1', 'images'))
    if rotation2:
        if not os.path.exists(os.path.join(path_save, 'Rotation2')):
            os.makedirs(os.path.join(path_save, 'Rotation2'))
        if not os.path.exists(os.path.join(path_save, 'Rotation2', 'masks')):
            os.makedirs(os.path.join(path_save, 'Rotation2', 'masks'))
        if not os.path.exists(os.path.join(path_save, 'Rotation2', 'images')):
            os.makedirs(os.path.join(path_save, 'Rotation2', 'images'))
    if rotation3:
        if not os.path.exists(os.path.join(path_save, 'Rotation3')):
            os.makedirs(os.path.join(path_save, 'Rotation3'))
        if not os.path.exists(os.path.join(path_save, 'Rotation3', 'masks')):
            os.makedirs(os.path.join(path_save, 'Rotation3', 'masks'))
        if not os.path.exists(os.path.join(path_save, 'Rotation3', 'images')):
            os.makedirs(os.path.join(path_save, 'Rotation3', 'images'))
    if rotation4:
        if not os.path.exists(os.path.join(path_save, 'Rotation4')):
            os.makedirs(os.path.join(path_save, 'Rotation4'))
        if not os.path.exists(os.path.join(path_save, 'Rotation4', 'masks')):
            os.makedirs(os.path.join(path_save, 'Rotation4', 'masks'))
        if not os.path.exists(os.path.join(path_save, 'Rotation4', 'images')):
            os.makedirs(os.path.join(path_save, 'Rotation4', 'images'))
    if rotation5:
        if not os.path.exists(os.path.join(path_save, 'Rotation5')):
            os.makedirs(os.path.join(path_save, 'Rotation5'))
        if not os.path.exists(os.path.join(path_save, 'Rotation5', 'masks')):
            os.makedirs(os.path.join(path_save, 'Rotation5', 'masks'))
        if not os.path.exists(os.path.join(path_save, 'Rotation5', 'images')):
            os.makedirs(os.path.join(path_save, 'Rotation5', 'images'))
    if rotation6:
        if not os.path.exists(os.path.join(path_save, 'Rotation6')):
            os.makedirs(os.path.join(path_save, 'Rotation6'))
        if not os.path.exists(os.path.join(path_save, 'Rotation6', 'masks')):
            os.makedirs(os.path.join(path_save, 'Rotation6', 'masks'))
        if not os.path.exists(os.path.join(path_save, 'Rotation6', 'images')):
            os.makedirs(os.path.join(path_save, 'Rotation6', 'images'))
    if rotation7:
        if not os.path.exists(os.path.join(path_save, 'Rotation7')):
            os.makedirs(os.path.join(path_save, 'Rotation7'))
        if not os.path.exists(os.path.join(path_save, 'Rotation7', 'masks')):
            os.makedirs(os.path.join(path_save, 'Rotation7', 'masks'))
        if not os.path.exists(os.path.join(path_save, 'Rotation7', 'images')):
            os.makedirs(os.path.join(path_save, 'Rotation7', 'images'))

    for k in range(len(true_imgs)):
        im = imageio.imread(true_masks[k])
        im_ = Image.fromarray(im)
        im_true = imageio.imread(true_imgs[k])
        im_true_ = Image.fromarray(im_true)

        if rotation1 != 0:
            tmp_im = im_.rotate(rotation1)
            imageio.imsave(os.path.join(path_save, 'Rotation1',
                                        'masks', str(k).zfill(10)+'.png'), np.array(tmp_im))
            tmp_im = im_true_.rotate(rotation1)
            imageio.imsave(os.path.join(path_save, 'Rotation1',
                                        'images', str(k).zfill(10)+'.png'), np.array(tmp_im))

        if rotation2 != 0:
            tmp_im = im_.rotate(rotation2)
            imageio.imsave(os.path.join(path_save, 'Rotation2',
                                        'masks', str(k).zfill(10)+'.png'), np.array(tmp_im))
            tmp_im = im_true_.rotate(rotation2)
            imageio.imsave(os.path.join(path_save, 'Rotation2',
                                        'images', str(k).zfill(10)+'.png'), np.array(tmp_im))

        if rotation3 != 0:
            tmp_im = im_.rotate(rotation3)
            imageio.imsave(os.path.join(path_save, 'Rotation3',
                                        'masks', str(k).zfill(10)+'.png'), np.array(tmp_im))
            tmp_im = im_true_.rotate(rotation3)
            imageio.imsave(os.path.join(path_save, 'Rotation3',
                                        'images', str(k).zfill(10)+'.png'), np.array(tmp_im))

        if k % 2 == 0:
            if rotation4 != 0:
                tmp_im = im_.rotate(rotation4)
                imageio.imsave(os.path.join(path_save, 'Rotation4', 'masks', str(
                    k).zfill(10)+'.png'), np.array(tmp_im))
                tmp_im = im_true_.rotate(rotation4)
                imageio.imsave(os.path.join(path_save, 'Rotation4', 'images', str(
                    k).zfill(10)+'.png'), np.array(tmp_im))

            if rotation5 != 0:
                tmp_im = im_.rotate(rotation5)
                imageio.imsave(os.path.join(path_save, 'Rotation5', 'masks', str(
                    k).zfill(10)+'.png'), np.array(tmp_im))
                tmp_im = im_true_.rotate(rotation5)
                imageio.imsave(os.path.join(path_save, 'Rotation5', 'images', str(
                    k).zfill(10)+'.png'), np.array(tmp_im))

        im = imageio.imread(true_masks[k_max-1-k])
        im_ = Image.fromarray(im)
        im_true = imageio.imread(true_imgs[k_max-1-k])
        im_true_ = Image.fromarray(im_true)
        if rotation6 != 0:
            tmp_im = im_.rotate(rotation6)
            imageio.imsave(os.path.join(path_save, 'Rotation6',
                                        'masks', str(k).zfill(10)+'.png'), np.array(tmp_im))
            tmp_im = im_true_.rotate(rotation6)
            imageio.imsave(os.path.join(path_save, 'Rotation6',
                                        'images', str(k).zfill(10)+'.png'), np.array(tmp_im))

        if rotation7 != 0:
            tmp_im = im_.rotate(rotation7)
            imageio.imsave(os.path.join(path_save, 'Rotation7',
                                        'masks', str(k).zfill(10)+'.png'), np.array(tmp_im))
            tmp_im = im_true_.rotate(rotation7)
            imageio.imsave(os.path.join(path_save, 'Rotation7',
                                        'images', str(k).zfill(10)+'.png'), np.array(tmp_im))
