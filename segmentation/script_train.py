from tensorflow.python.client import device_lib
from keras.losses import binary_crossentropy
import keras.optimizers as optimizers
from keras.models import load_model
import numpy as np
import shutil
import keras
import os

import data_loader
import train
import model

keras.backend.set_image_data_format('channels_last')

"""
TODO:
 - If test images empty: make sure it wotks.
"""

MODE = "GPU" if "GPU" in [k.device_type for k in device_lib.list_local_devices()] else "CPU"
print(MODE)

"""
This script is a demo of how to use the diffrent functions from https://github.com/lebrat/Biolapse to train a neural network
to segment images with temporal information. 

Parameters:
 - epoch: number of iteration in learning phase.
 - lr: learning rate, step-size of the optimization algorithm.
 - momentum: weight of previous iteration.
 - decay: weight of the penalization of the weights. Tends to have more small value weights.
 - steps_per_epoch: number of iteration within an iteration.
 - batch_size: number of sample used to aggregate the descent step in optimization method. If GPU
  runs out of memory, this might be because batch_size is too large.
 - type_im: np.uint8 or np.uint16. Used to properly load images.
 - (nx, ny): spatial shape of images used to train neural network. If GPU runs out of memory, this might
  be because nx and/or ny is too large.
 - TIME: size of the temporal sample use to predict segmentation. Large value of TIME might lead to better
 results but it requires GPU with large memory.
 - path_train, path_test: folders containin training and testing images. Can contain many inside folders, 
 at the end masks should be in a folder name 'masks' and images in a folder 'images' at the same level.
 - model: 'Unet3D', 'Unet2D' or 'LSTM' neural network model.

Output:
 Save in Data/Model/Model a file name_save+'.h5' containing the neural network. Validation and training
  informations are stored in Data/Model/Information/name_save+'.p'. 
"""

## Parameters
epoch = 100
lr = 1e-2
momentum = 0.8
decay = 1e-6
steps_per_epoch = 100
batch_size = 8
type_im = np.uint16
nx = ny = 128
TIME = 10

path_train = '../Data/Segmentation/Train'
path_test = '../Data/Segmentation/Test'

name_save = 'nn_unet2D'
model_name = 'Unet2D' # 'LSTM'

## Augmentation
# Uncomment to use data augmentation, need to specify the path of your images
# path_imgs = 'path/of/train/images/*.png'
# path_masks = 'path/of/train/masks/*.png'
# path_save = '../Data/Segmentation/Train'
# data_loader.augmentation_png(path_imgs,path_masks,path_save, original=True, oneOfTwo=True, reverse=True, reverseOneOfTwo=True,
#  rotation1=30, rotation2=150,rotation3=240,rotation4=90,rotation5=290,rotation6=120, rotation7=330)

# path_imgs = 'path/of/test/images/*.png'
# path_masks = 'path/of/test/masks/*.png'
# path_save = '../Data/Segmentation/Test'
# data_loader.augmentation_png(path_imgs,path_masks,path_save, original=True, oneOfTwo=True, reverse=True, reverseOneOfTwo=True,
#  rotation1=30, rotation2=150,rotation3=240,rotation4=90,rotation5=290,rotation6=120, rotation7=330)

## Load data and model
if model_name=='Unet3D':
    # Data
    X_train, Y_train = data_loader.path_to_time_batchs(path_train,nx,ny,TIME=TIME,type_im=type_im,format_im='png',
    code_im1='images',code_im2='masks')
    X_train = np.expand_dims(np.array(X_train, dtype=np.float32),4)
    Y_train = np.expand_dims(np.array(Y_train, dtype=np.float32),4)
    for t in range(X_train.shape[0]):
        if np.max(X_train[t])!=0:
            X_train[t] = X_train[t]/np.max(X_train[t])
        if np.max(Y_train[t])!=0:
            Y_train[t] = Y_train[t]/np.max(Y_train[t])
    X_test, Y_test= data_loader.path_to_time_batchs(path_test,nx,ny,TIME=TIME,type_im=type_im,format_im='png',
    code_im1='images',code_im2='masks')
    X_test = np.expand_dims(np.array(X_test, dtype=np.float32),4)
    Y_test = np.expand_dims(np.array(Y_test, dtype=np.float32),4)
    for t in range(X_test.shape[0]):
        if np.max(X_test[t])!=0:
            X_test[t] = X_test[t]/np.max(X_test[t])
        if np.max(Y_test[t])!=0:
            Y_test[t] = Y_test[t]/np.max(Y_test[t])
    List_id_train = data_loader.array_to_npy(X_train,Y_train,test=False,name=name_save)
    List_id_test = data_loader.array_to_npy(X_test,Y_test,test=True,name=name_save)
    params = {'dim': (nx,ny,TIME,1),
            'batch_size': batch_size,
            'n_channels': 1,
            'shuffle': True,
            'name' : name_save}
    generator_train = data_loader.DataGenerator3D(List_id_train, **params)
    generator_test = data_loader.DataGenerator3D(List_id_test, **params)
    # Model
    # Change model parameters here
    net = model.unet_model_3d(input_shape=(nx,ny,TIME,1), pool_size=(2, 2, 1), n_labels=1, deconvolution=False,
                    depth=4, n_base_filters=16, include_label_wise_dice_coefficients=False, metrics='mse',
                    batch_normalization=True, activation_name="sigmoid")
elif model_name=='LSTM':
    # Data
    X_train, Y_train = data_loader.path_to_time_batchs(path_train,nx,ny,TIME=TIME,type_im=type_im,format_im='png',
    code_im1='images',code_im2='masks')
    X_train = np.expand_dims(np.array(X_train, dtype=np.float32),4)
    Y_train = np.expand_dims(np.array(Y_train, dtype=np.float32),4)
    for t in range(X_train.shape[0]):
        X_train[t] = X_train[t]/np.max(X_train[t])
        Y_train[t] = Y_train[t]/np.max(Y_train[t])
    X_test, Y_test= data_loader.path_to_time_batchs(path_test,nx,ny,TIME=TIME,type_im=type_im,format_im='png',
    code_im1='images',code_im2='masks')
    X_test = np.expand_dims(np.array(X_test, dtype=np.float32),4)
    Y_test = np.expand_dims(np.array(Y_test, dtype=np.float32),4)
    for t in range(X_test.shape[0]):
        X_test[t] = X_test[t]/np.max(X_test[t])
        Y_test[t] = Y_test[t]/np.max(Y_test[t])
    X_train = np.rollaxis(X_train,3,1)
    Y_train = np.rollaxis(Y_train,3,1)
    X_test = np.rollaxis(X_test,3,1)
    Y_test = np.rollaxis(Y_test,3,1)
    List_id_train = data_loader.array_to_npy(X_train,Y_train,test=False,name=name_save)
    List_id_test = data_loader.array_to_npy(X_test,Y_test,test=True,name=name_save)
    params = {'dim': (TIME,nx,ny,1),
            'batch_size': batch_size,
            'n_channels': 1,
            'shuffle': True,
            'name' : name_save}
    generator_train = data_loader.DataGenerator3D(List_id_train, **params)
    generator_test = data_loader.DataGenerator3D(List_id_test, **params)
    # Model
    # Change model parameters here
    net = model.LSTMNET3((TIME,nx,ny,1),filters=16)
elif model_name=='Unet2D':
    # Data
    X_train, Y_train = data_loader.path_to_batchsV2(path_train,nx,ny,type_im=type_im,format_im='png',
    code_im1='images',code_im2='masks')
    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.float32)
    for t in range(X_train.shape[0]):
        if np.max(X_train[t])!=0:
            X_train[t] = X_train[t]/np.max(X_train[t])
        if np.max(Y_train[t])!=0:
            Y_train[t] = Y_train[t]/np.max(Y_train[t])
    X_test, Y_test = data_loader.path_to_batchsV2(path_test,nx,ny,type_im=type_im,format_im='png',
    code_im1='images',code_im2='masks')
    X_test = np.array(X_test, dtype=np.float32)
    Y_test = np.array(Y_test, dtype=np.float32)
    for t in range(X_test.shape[0]):
        if np.max(X_test[t])!=0:
            X_test[t] = X_test[t]/np.max(X_test[t])
        if np.max(Y_test[t])!=0:
            Y_test[t] = Y_test[t]/np.max(Y_test[t])
    List_id_train = data_loader.array_to_npy(X_train,Y_train,test=False,name=name_save)
    List_id_test = data_loader.array_to_npy(X_test,Y_test,test=True,name=name_save)
    params = {'dim': (nx,ny,1),
            'batch_size': batch_size,
            'n_channels': 1,
            'shuffle': True,
            'name' : name_save}
    generator_train = data_loader.DataGenerator3D(List_id_train, **params)
    generator_test = data_loader.DataGenerator3D(List_id_test, **params)
    # Model
    # Change model parameters here
    # net = model.Unet2D(input_size = (nx,ny,1),filters=16)
    net = model.Unet2DBowl(nx,ny)
else:
    raise ValueError('Model name not understood: {0}. Must be in {{Unet3D, Unet2D, LSTM}}.'.format(model_name))
net.summary()

# Optimization
opt = optimizers.SGD(lr=lr,decay=decay,momentum=momentum)
# net.compile(optimizer=opt, loss=binary_crossentropy,metrics=['mse', 'acc'])
net.compile(optimizer='adam', loss=train.bce_dice_loss,metrics=[train.mean_iou])


def step_decay(epoch):
    initial_lrate = lr
    drop = 0.5
    epochs_drop = 15.0
    lrate = initial_lrate * np.math.pow(drop, np.floor((1+epoch)/epochs_drop))
    return lrate
## Training
net = train.training_generator(net,generator_train,epochs=epoch,steps=steps_per_epoch,
    save_name=name_save,generator_validation=generator_test,step_decay=step_decay)

shutil.rmtree(os.path.join(os.getcwd(),'tmp'),ignore_errors=True)