from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
import pickle
import os
import numpy as np
import time

"""
Train model with generator.
Save weights in Data/Model during iteration. The final model is saved in Data/Model.
"""
def training_generator(model,generator,epochs=1,steps=50,save_name='default',generator_validation=[],step_decay=[]):
    t_start = time.time()
    if not os.path.exists(os.path.join(os.getcwd(),'..','Data','Segmentation','Information')):
        os.makedirs(os.path.join(os.getcwd(),'..','Data','Segmentation','Information'))
    if not os.path.exists(os.path.join(os.getcwd(),'..','Data','Segmentation','Model')):
        os.makedirs(os.path.join(os.getcwd(),'..','Data','Segmentation','Model'))

    checkpoint = ModelCheckpoint(os.path.join(os.getcwd(),'..','Data','Segmentation','Model',save_name+'best.hdf5'), monitor='loss', verbose=1, save_best_only=True, mode='auto')
    # earlystopper = EarlyStopping(patience=50, verbose=1)
    tsboard = TensorBoard(log_dir=os.path.join('tmp','tb'))
    if step_decay ==[]:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        patience=5, min_lr=0.00001)
    else:
        reduce_lr = LearningRateScheduler(step_decay)
    if generator_validation!=[]:
        history=model.fit_generator(generator=generator,
                        validation_data=generator_validation,
                        validation_steps=20,
                        use_multiprocessing=True,
                        steps_per_epoch=steps,nb_epoch=epochs,verbose=1,
                        callbacks=[checkpoint,tsboard,reduce_lr])
    else:
        history=model.fit_generator(generator=generator,
                    use_multiprocessing=True,
                    steps_per_epoch=steps,nb_epoch=epochs,verbose=1,
                    callbacks=[checkpoint,tsboard,reduce_lr])

    elapsed_time = time.time() - t_start
    print("Training time: ",elapsed_time)
    with open(os.path.join(os.getcwd(),'..','Data','Segmentation','Information',save_name+'plots.p'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    model.save(os.path.join(os.getcwd(),'..','Data','Segmentation','Model',save_name+'.h5'))
    np.save(os.path.join(os.getcwd(),'..','Data','Segmentation','Model',save_name+'time.npy'),elapsed_time)
    return model

"""
Train model with array as dataset.
Save weights in Data/Model during iteration. The final model is saved in Data/Model.
"""
def train_model_array(model,x_train,y_train,batch_size=32,epochs=1,save_name='default',x_test=[],y_test=[],step_decay=[]):
    t_start = time.time()
    if not os.path.exists(os.path.join(os.getcwd(),'..','Data','Segmentation','Information')):
        os.makedirs(os.path.join(os.getcwd(),'..','Data','Segmentation','Information'))
    if not os.path.exists(os.path.join(os.getcwd(),'..','Data','Segmentation','Model')):
        os.makedirs(os.path.join(os.getcwd(),'..','Data','Segmentation','Model'))
    if step_decay ==[]:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
        patience=5, min_lr=0.00001)
    else:
        reduce_lr = LearningRateScheduler(step_decay)

    checkpoint = ModelCheckpoint(os.path.join(os.getcwd(),'..','Data','Segmentation','Model',save_name+'best.hdf5'), monitor='loss', verbose=1, save_best_only=True, mode='auto')
    if x_test!=[]:
        history=model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=epochs,
          callbacks=[checkpoint,TensorBoard(log_dir='/tmp/tb'),reduce_lr],
          validation_data=(x_test,y_test))
    else:
        history=model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=epochs,
          callbacks=[checkpoint,TensorBoard(log_dir='/tmp/tb'),reduce_lr])
    # ,LearningRateScheduler(step_decay)])
    
    elapsed_time = time.time() - t_start
    print("Training time: ",elapsed_time)
    with open(os.path.join(os.getcwd(),'..','Data','Segmentation','Information',save_name+'plots.p'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    model.save(os.path.join(os.getcwd(),'..','Data','Segmentation','Model',save_name+'.h5'))
    np.save(os.path.join(os.getcwd(),'..','Data','Segmentation','Model',save_name+'time.npy'),elapsed_time)
    return model

# Import all the necessary libraries
import os
import datetime
import glob
import random
import sys

import matplotlib.pyplot as plt
import skimage.io                                     #Used for imshow function
import skimage.transform                              #Used for resize function
from skimage.morphology import label                  #Used for Run-Length-Encoding RLE to create final submission

import numpy as np
import pandas as pd

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
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
# Custom IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)