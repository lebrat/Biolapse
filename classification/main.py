from train_phase_predictor import train_phase_predictor
from extract_data_train import extract_data
import numpy as np
import os 

'''
This script must be run to train a neural network to classify phase of cells.
'''


## Parameters
path_to_crop=os.path.join('..','Data','Outputs')
nx = ny = 128
nclass = 4 # G, erly S, mid S, late S
seq_test = 0 # number of the test sequence
save_png = True

## Extrat data for training
extract_data(path_to_crop,nx,ny,nclass,seq_test,save_png)

## Load data
im_train = np.load(os.path.join('..','Data','Classification','im_train.npy'))
im_test = np.load(os.path.join('..','Data','Classification','im_test.npy'))
feature_train = np.load(os.path.join('..','Data','Classification','feature_train.npy'))
feature_test = np.load(os.path.join('..','Data','Classification','feature_test.npy'))

## Train neural network for phase prediction
train_phase_predictor(im_train,feature_train,im_test,feature_test,save_png,epochs=20,nit_train=100,nbatch=30)
