from keras.models import load_model, Model
import numpy as np
import data_loader
import matplotlib.pyplot as plt



## Parameters
type_im = np.uint16
nx = ny = 128
img_width=img_height=nx

path_test = '../Data/Segmentation/Test'
path_test = '../Data/Segmentation/Test/augmentation/SUM_PCNA_74_3Dguassianblur/Original/'

# name_save = 'nn_unet2D'
# model_name = 'Unet2D' # 'LSTM'

# Data
X_test, Y_test = data_loader.path_to_batchs(path_test,256,256,type_im=type_im,format_im='png',
code_im1='images',code_im2='masks')
X_test = np.array(X_test, dtype=np.float32)
Y_test = np.array(Y_test, dtype=np.float32)
for t in range(X_test.shape[0]):
    if np.max(X_test[t])!=0:
        X_test[t] = X_test[t]/np.max(X_test[t])
    if np.max(Y_test[t])!=0:
        Y_test[t] = Y_test[t]/np.max(Y_test[t])



import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import add, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model, plot_model
from keras import backend as K
# Design our model architecture here
def keras_model(img_width=256, img_height=256):
    '''
    Modified from https://keunwoochoi.wordpress.com/2017/10/11/u-net-on-keras-2-0/
    '''
    n_ch_exps = [4, 5, 6, 7, 8, 9]   #the n-th deep channel's exponent i.e. 2**n 16,32,64,128,256
    k_size = (3, 3)                  #size of filter kernel
    k_init = 'he_normal'             #kernel initializer

    if K.image_data_format() == 'channels_first':
        ch_axis = 1
        input_shape = (3, img_width, img_height)
    elif K.image_data_format() == 'channels_last':
        ch_axis = 3
        input_shape = (img_width, img_height, 1)

    inp = Input(shape=input_shape)
    encodeds = []

    # encoder
    enc = inp
    print(n_ch_exps)
    for l_idx, n_ch in enumerate(n_ch_exps):
        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        enc = Dropout(0.1*l_idx,)(enc)
        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        encodeds.append(enc)
        #print(l_idx, enc)
        if n_ch < n_ch_exps[-1]:  #do not run max pooling on the last encoding/downsampling step
            enc = MaxPooling2D(pool_size=(2,2))(enc)
    
    # decoder
    dec = enc
    print(n_ch_exps[::-1][1:])
    decoder_n_chs = n_ch_exps[::-1][1:]
    for l_idx, n_ch in enumerate(decoder_n_chs):
        l_idx_rev = len(n_ch_exps) - l_idx - 2  #
        dec = Conv2DTranspose(filters=2**n_ch, kernel_size=k_size, strides=(2,2), activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = concatenate([dec, encodeds[l_idx_rev]], axis=ch_axis)
        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = Dropout(0.1*l_idx)(dec)
        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)

    outp = Conv2DTranspose(filters=1, kernel_size=k_size, activation='sigmoid', padding='same', kernel_initializer='glorot_normal')(dec)

    model = Model(inputs=[inp], outputs=[outp])
    
    return model


def reconstruction(model,imgs,nx=128,ny=128,thresh=0.5):
    from skimage.transform import resize
    im_list=[]
    mask_list=[]
    for k in range(imgs.shape[0]):
        im=imgs[k]
        n1=im.shape[0]
        n2=im.shape[1]
        if n1//nx==0 or n2//ny==0:
            img = np.array(resize(im, (nx, ny), mode='constant',
                        preserve_range=True),dtype=np.float32)
            img=(img-np.min(img))/(np.max(img)-np.min(img))
            img=np.expand_dims(img,0)
            if len(img.shape)==3:
                img=np.expand_dims(img,3)
            preds_val = model.predict(img, verbose=1)
            preds_val_t = np.squeeze(preds_val > thresh)
            im_list.append(im)
            tmp=resize(preds_val_t, (n1, n2), mode='constant',
                        preserve_range=True)
            mask_list.append(tmp)
        else:
            im_save=np.zeros((im.shape[0],im.shape[1]))
            mask=np.zeros((im.shape[0],im.shape[1]))
            for kx in range(n1//nx):
                for ky in range(n2//ny):
                    img=im[kx*nx:(kx+1)*nx,ky*ny:(ky+1)*ny,:]
                    img=np.expand_dims(img,0)
                    preds_val = model.predict(img, verbose=1)
                    preds_val_t = (preds_val > thresh)
                    mask[kx*nx:(kx+1)*nx,ky*ny:(ky+1)*ny]=np.squeeze(preds_val_t)
                    im_save[kx*nx:(kx+1)*nx,ky*ny:(ky+1)*ny]=np.squeeze(img)
            im_list.append(im_save)
            mask_list.append(mask)
    return im_list,mask_list


# Reload the model
model = keras_model(img_width=img_width, img_height=img_height)
model.load_weights("model-weights.hdf5")

im_list,mask_list=reconstruction(model,X_test[:100],nx,ny,thresh=0.5)
mask=np.array(mask_list)
imgs=np.array(im_list)

# Define IoU metric as a regular function, to manually check result
def cal_iou(A, B):
    intersection = np.logical_and(A, B)
    union = np.logical_or(A, B)
    iou = np.sum(intersection > 0) / np.sum(union > 0)
    return iou
# calcualte average iou of validation images, the result from tensorflow seems too high. 
iou=[]
for i in range(mask.shape[0]):
    iou.append(cal_iou(np.squeeze(Y_test[i]), np.squeeze(mask[i])))
print('Average Validate IOU: {}'.format(round(np.mean(iou),2)))
#plt.figure(figsize=(20,10.5))
plt.figure(figsize=(20,16))
x, y = 16,3
for i in range(y):  
    for j in range(x):
        if i*x+j>=X_test.shape[0]:
            break
        # train image
        plt.subplot(y*3, x, i*3*x+j+1)
        pos = i*x+j
        plt.imshow(np.squeeze(X_test[pos]))
        # plt.title('Image #{}\nIOU {}'.format(pos,round(cal_iou(np.squeeze(Y_test[pos]), np.squeeze(mask[pos])),2)))
        plt.axis('off')
        plt.subplot(y*3, x, (i*3+1)*x+j+1)
        plt.imshow(np.squeeze(Y_test[pos]))
        # plt.title('Mask')
        plt.axis('off')
        plt.subplot(y*3, x, (i*3+2)*x+j+1)
        plt.imshow(np.squeeze(mask[pos]))
        # plt.title('Predict')
        plt.axis('off')
    if i*x+j>=X_test.shape[0]:
        break
plt.show()

import imageio
import os
if not os.path.exists(os.path.join('Outputs','Unet2DCrop')):
    os.makedirs(os.path.join('Outputs','Unet2DCrop'))
for l in range(imgs.shape[0]):
    tmp=imgs[l]
    tmp=np.array(255*((tmp-np.min(tmp))/(np.max(tmp)-np.min(tmp))),dtype=np.uint8)
    imageio.imsave(os.path.join('Outputs','Unet2DCrop')+'/original_'+str(l).zfill(7)+'.png', tmp)
    tmp=mask[l]
    tmp=np.array(255*((tmp-np.min(tmp))/(np.max(tmp)-np.min(tmp))),dtype=np.uint8)
    imageio.imsave(os.path.join('Outputs','Unet2DCrop')+'/mask_'+str(l).zfill(7)+'.png', tmp)
