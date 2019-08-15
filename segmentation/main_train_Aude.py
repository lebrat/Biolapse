from tensorflow.python.client import device_lib
from keras.models import load_model
import keras
from keras.losses import binary_crossentropy
import keras.optimizers as optimizers
import losses

import numpy as np
import data_loader
import train
# import visualize
import model
# close = lambda :visualize.close_all()
keras.backend.set_image_data_format('channels_last')

"""
TODO:
 - Dice coefficients as validation (training?)
 - Dataset de validation -> compute several indexes as postprocessing
 - Interface d'augmentation
 - tiff_to_png: garde rstructure des dossier dans le dossier png, non juste dernier 
"""


MODE = "GPU" if "GPU" in [k.device_type for k in device_lib.list_local_devices()] else "CPU"
print(MODE)

## Parameters
# get parser?
epoch = 75
lr = 1e-1
momentum = 0.8
decay = 1e-6
steps_per_epoch = 10
batch_size = 1
type_im = np.uint16
nx = ny = 256
TIME = 10

## Generator
path_train = 'AudeGuenole/Train'
path_test = 'AudeGuenole/Test'


## Model
# Unet 3D
name = 'nn_Aude_Unet3D_256_old_train'
X_train, Y_train = data_loader.path_to_time_batchs(path_train,nx,ny,TIME=TIME,type_im=type_im,format_im='png',
	code_im1='images',code_im2='masks')
X_train = np.expand_dims(np.array(X_train, dtype=np.float32)/np.iinfo(type_im).max,4)
Y_train = np.expand_dims(np.array(Y_train, dtype=np.float32)/np.iinfo(type_im).max,4)
List_id_train = data_loader.array_to_npy(X_train,Y_train,test=False,name=name)

X_test, Y_test= data_loader.path_to_time_batchs(path_test,nx,ny,TIME=TIME,type_im=type_im,format_im='png',
	code_im1='images',code_im2='masks')
X_test = np.expand_dims(np.array(X_test, dtype=np.float32)/np.iinfo(type_im).max,4)
Y_test = np.expand_dims(np.array(Y_test, dtype=np.float32)/np.iinfo(type_im).max,4)
List_id_test = data_loader.array_to_npy(X_test,Y_test,test=False,name=name)

params = {'dim': (nx,ny,TIME,1),
          'batch_size': batch_size,
          'n_channels': 1,
          'shuffle': True,
          'name' : name}
generator_train = data_loader.DataGenerator3D(List_id_train, **params)
generator_test = data_loader.DataGenerator3D(List_id_test, **params)

net = model.unet_model_3d(input_shape=(nx,ny,TIME,1), pool_size=(2, 2, 1), n_labels=1, deconvolution=False,
                  depth=4, n_base_filters=16, include_label_wise_dice_coefficients=False, metrics='mse',
                  batch_normalization=True, activation_name="sigmoid")
net.summary()
## Optimization
# loss = binary_crossentropy
# opt = optimizers.SGD(lr=lr,decay=decay,momentum=momentum)
# net.compile(optimizer=opt, loss=loss,metrics=['mse', 'acc'])

l1 = 1.
l2 = -0.1
loss1 = binary_crossentropy
loss2 = losses.dice_coef_loss_K
def custom_loss(y_true,y_pred):
    return l1*loss1(y_true,y_pred)+l2*loss2(y_true,y_pred)

opt = optimizers.SGD(lr=lr,decay=decay,momentum=momentum)
# net.compile(optimizer=opt, loss=loss,metrics=['mse', 'acc'])
net.compile(optimizer=opt, loss = custom_loss, metrics=['mse', 'acc'])

## Training
net = train.training_generator(net,generator_train,epochs=epoch,steps=steps_per_epoch,
	save_name=name,generator_validation=generator_test)


# LSTM
# name = 'nn_Aude_LSTM'
# X_train, Y_train = data_loader.path_to_time_batchs(path_train,nx,ny,TIME=TIME,type_im=type_im,format_im='png',
# 	code_im1='images',code_im2='masks')
# X_test, Y_test= data_loader.path_to_time_batchs(path_test,nx,ny,TIME=TIME,type_im=type_im,format_im='png',
# 	code_im1='images',code_im2='masks')
# net = model.LSTMNET3((TIME,nx,ny,1),filters=16)
# X_train = np.expand_dims(np.array(X_train, dtype=np.float32)/np.iinfo(type_im).max,4)
# Y_train = np.expand_dims(np.array(Y_train, dtype=np.float32)/np.iinfo(type_im).max,4)
# X_test = np.expand_dims(np.array(X_test, dtype=np.float32)/np.iinfo(type_im).max,4)
# Y_test = np.expand_dims(np.array(Y_test, dtype=np.float32)/np.iinfo(type_im).max,4)
# X_train = np.rollaxis(X_train,3,1)
# Y_train = np.rollaxis(Y_train,3,1)
# X_test = np.rollaxis(X_test,3,1)
# Y_test = np.rollaxis(Y_test,3,1)
# List_id_train = data_loader.array_to_npy(X_train,Y_train,test=False,name=name)
# List_id_test = data_loader.array_to_npy(X_test,Y_test,test=False,name=name)
# params = {'dim': (TIME,nx,ny,1),
#           'batch_size': batch_size,
#           'n_channels': 1,
#           'shuffle': True,
#           'name' : name}
# generator_train = data_loader.DataGenerator3D(List_id_train, **params)
# generator_test = data_loader.DataGenerator3D(List_id_test, **params)
# net.summary()
# ## Optimization
# loss = binary_crossentropy
# opt = optimizers.SGD(lr=lr,decay=decay,momentum=momentum)
# net.compile(optimizer=opt, loss=loss,metrics=['mse', 'acc'])
# ## Training
# net = train.training_generator(net,generator_train,epochs=epoch,steps=steps_per_epoch,
# 	save_name=name,generator_validation=generator_test)


# # # Unet 2D
# name = 'nn_Aude_Unet2D'
# X_train, Y_train = data_loader.path_to_time_batchs(path_train,nx,ny,TIME=1,type_im=type_im,format_im='png',
# 	code_im1='images',code_im2='masks')
# X_test, Y_test= data_loader.path_to_time_batchs(path_test,nx,ny,TIME=1,type_im=type_im,format_im='png',
# 	code_im1='images',code_im2='masks')

# X_train = np.array(X_train, dtype=np.float32)/np.iinfo(type_im).max
# Y_train = np.array(Y_train, dtype=np.float32)/np.iinfo(type_im).max
# X_test = np.array(X_test, dtype=np.float32)/np.iinfo(type_im).max
# Y_test = np.array(Y_test, dtype=np.float32)/np.iinfo(type_im).max
# List_id_train = data_loader.array_to_npy(X_train,Y_train,test=False,name=name)
# List_id_test = data_loader.array_to_npy(X_test,Y_test,test=False,name=name)
# params = {'dim': (nx,ny,1),
#           'batch_size': batch_size,
#           'n_channels': 1,
#           'shuffle': True,
#           'name' : name}
# generator_train = data_loader.DataGenerator3D(List_id_train, **params)
# generator_test = data_loader.DataGenerator3D(List_id_test, **params)
# net = model.Unet2D(input_size = (nx,ny,1),filters=64)
# net.summary()
# ## Optimization
# loss = binary_crossentropy
# opt = optimizers.SGD(lr=lr,decay=decay,momentum=momentum)
# net.compile(optimizer=opt, loss=loss,metrics=['mse', 'acc'])
# ## Training
# net = train.training_generator(net,generator_train,epochs=epoch,steps=steps_per_epoch,
# 	save_name=name,generator_validation=generator_test)



