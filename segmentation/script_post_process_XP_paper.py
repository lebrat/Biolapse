from tensorflow.python.client import device_lib
from keras.models import load_model
import numpy as np
import data_loader
import visualize
import losses
import pylab
import time
import utils

## Parameters
name = 'nn_demo'
batch_size = 2
type_im = np.uint16
type_tf = np.float32

thresh = 0.2
SAVE = True
VIDEO = True

model_name = 'Unet3D'
# path_test = '../Data/Segmentation/Test/augmentation/SUM_PCNA_74_3Dguassianblur/Original'
path_test2D = '../Data/Segmentation/Test2D'
path_test3D = '../Data/Segmentation/Test3D'

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
MODE = "GPU" if "GPU" in [k.device_type for k in device_lib.list_local_devices()] else "CPU"
print(MODE)

name_save_list = ['nn_XP_Unet3D','nn_XP_Unet2D','nn_XP_LSTM','nn_Unet2D_bowl_crop']
model_name_list = ['Unet3D', 'Unet2D', 'LSTM','Unet2D_bowl_crop']
# name_save_list = ['nn_Unet2D_bowl_crop']
# model_name_list = ['Unet2D_bowl_crop']

for i in range(len(model_name_list)):
    model_name = model_name_list[i]
    name_save = name_save_list[i]

    ## Load net
    path_h5 = os.path.join(os.getcwd(),'..','Data','Segmentation','Model',name_save+'.h5')
    net = load_model(path_h5,custom_objects={'bce_dice_loss': utils.bce_dice_loss, 'mean_iou': utils.mean_iou})
    # net = load_model(path_h5)
    net.summary()

    ## Load data
    load_data = False
    if model_name =='Unet3D':
        nx = net.input.shape[1].value
        ny = net.input.shape[2].value
        TIME = net.input.shape[3].value
        X_test, Y_test= data_loader.path_to_time_batchs(path_test3D,nx,ny,TIME=TIME,type_im=type_im,format_im='png',
            code_im1='images',code_im2='masks')
        X_test = np.expand_dims(np.array(X_test, dtype=np.float32)/np.iinfo(type_im).max,4)
        Y_test = np.expand_dims(np.array(Y_test, dtype=np.float32)/np.iinfo(type_im).max,4)
        if SAVE:
            y_pred = visualize.save_png_list(net, X_test, name=name_save, thresh=thresh, format_im=type_im) 
        # Display an image
        visualize.display_temporal_output(net,X_test,N=1,thresh=thresh)
        load_data = True
    if model_name =='Unet2D':
        nx = net.input.shape[1].value
        ny = net.input.shape[2].value
        X_test, Y_test= data_loader.path_to_batchs(path_test2D,nx,ny,type_im=type_im,format_im='png',
            code_im1='images',code_im2='masks')
        if SAVE:
            y_pred = visualize.save_png_list(net, X_test, name=name_save, thresh=thresh, format_im=type_im) 
        # Display an image
        visualize.display_output(net,X_test,N=1,thresh=thresh)
        load_data = True
    elif model_name =='LSTM':
        nx = net.input.shape[2].value
        ny = net.input.shape[3].value
        TIME = net.input.shape[1].value
        X_test, Y_test= data_loader.path_to_time_batchs(path_test3D,nx,ny,TIME=TIME,type_im=type_im,format_im='png',
            code_im1='images',code_im2='masks')
        X_test = np.expand_dims(np.array(X_test, dtype=np.float32)/np.iinfo(type_im).max,4)
        Y_test = np.expand_dims(np.array(Y_test, dtype=np.float32)/np.iinfo(type_im).max,4)
        X_test = np.rollaxis(X_test,3,1)
        Y_test = np.rollaxis(Y_test,3,1)
        if SAVE:
            y_pred = visualize.save_png_list_channel_first(net, X_test, name=name_save, thresh=thresh, format_im=type_im)
        load_data = True  
    elif model_name=='Unet2D_bowl_crop':
        nx = net.input.shape[1].value
        ny = net.input.shape[2].value
        X_test, Y_test = data_loader.path_to_batchs_crop(path_test2D,nx,ny,type_im=type_im,format_im='png',
        code_im1='images',code_im2='masks')
        X_test = np.array(X_test, dtype=np.float32)
        Y_test = np.array(Y_test, dtype=np.float32)
        for t in range(X_test.shape[0]):
            if np.max(X_test[t])!=0:
                X_test[t] = X_test[t]/np.max(X_test[t])
            if np.max(Y_test[t])!=0:
                Y_test[t] = Y_test[t]/np.max(Y_test[t])
        if SAVE:
            y_pred = visualize.save_png_list(net, X_test, name=name_save, thresh=thresh, format_im=type_im)
    elif load_data==False:
        raise ValueError('Model name not understood: {0}. Must be in {{Unet3D, Unet2D, LSTM}}.'.format(model_name))

    # Display training inforamtions
    visualize.display_loss(name_save,save=True,custom=True)

    # Make video
    if VIDEO and not(model_name=='Unet2D'):
        visualize.save_png_video(net, X_test, thresh=thresh, format_im=type_im,name=name_save+'_demo_video')
        visualize.automatic_process(name_save+'_demo_video',cut=10,nb_movie=2,nb_seq_in_video=130)

    ## Compute score on test
    predict = net.predict(X_test)

    best_dice_score = -1e18
    best_jaccard_score = -1e18
    best_acc = -1e18
    best_score = -1e18
    best_thresh = -1
    # Find thresh that maximizes the average score
    for thresh in np.linspace(0.05,0.95,0.9/0.05+1):
        t_pred = time.time()
        y_pred = np.array(predict>thresh,np.uint8).flatten()
        t_pred = time.time() - t_pred

        Y_test = np.array(Y_test>0.5,dtype=np.uint8)
        dice_score = losses.dice_coef(Y_test.flatten(), y_pred.flatten())
        jaccard_score = losses.jaccard_coef(Y_test.flatten(), y_pred.flatten())
        acc = losses.accuracy(Y_test.flatten(), y_pred.flatten())
        score = dice_score/3+jaccard_score/3+acc/3

        if score > best_score:
            best_dice_score = dice_score
            best_jaccard_score = jaccard_score
            best_acc = acc
            best_score = score
            best_thresh= thresh
    print("Threshold: {0}".format(best_thresh))
    print("Prediction done in {0} seconds for {1} frames: {2} frame per second.".format(t_pred,X_test.shape[0],X_test.shape[0]/t_pred))
    print("Dice score: {0}. Jaccard score: {1}. Accuracy {2}.".format(best_dice_score,best_jaccard_score,best_acc))
