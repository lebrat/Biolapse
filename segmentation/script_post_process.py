from keras.models import load_model
import numpy as np
import data_loader
import visualize
import losses
import pylab
import time

## Parameters
name = 'nn_demo'
batch_size = 2
type_im = np.uint16
type_tf = np.float32

thresh = 0.2
SAVE = False
VIDEO = True

model_name = 'Unet3D'
path_test = 'Data/Test'

## Load net
path_h5 = 'Data/Model/'+name+'.h5'
net = load_model(path_h5)
net.summary()

## Load data
if model_name =='Unet3D':
    nx = net.input.shape[1].value
    ny = net.input.shape[2].value
    TIME = net.input.shape[3].value
    X_test, Y_test= data_loader.path_to_time_batchs(path_test,nx,ny,TIME=TIME,type_im=type_im,format_im='png',
        code_im1='images',code_im2='masks')
    X_test = np.expand_dims(np.array(X_test, dtype=np.float32)/np.iinfo(type_im).max,4)
    Y_test = np.expand_dims(np.array(Y_test, dtype=np.float32)/np.iinfo(type_im).max,4)
    if SAVE:
        y_pred = visualize.save_png_list(net, X_test, name=name, tresh=tresh, format_im=type_im) 
    # Display an image
    visualize.display_temporal_output(net,X_test,N=1,tresh=tresh)
if model_name =='Unet2D':
    nx = net.input.shape[1].value
    ny = net.input.shape[2].value
    X_test, Y_test= data_loader.path_to_batchs(path_test,nx,ny,type_im=type_im,format_im='png',
        code_im1='images',code_im2='masks')
    if SAVE:
        y_pred = visualize.save_png_list(net, X_test, name=name, tresh=tresh, format_im=type_im) 
    # Display an image
    visualize.display_output(net,X_test,N=1,tresh=tresh)

elif model_name =='LSTM':
    X_test = np.rollaxis(X_test,3,1)
    Y_test = np.rollaxis(Y_test,3,1)
    if SAVE:
        y_pred = visualize.save_png_list_channel_first(net, X_test, name=name, tresh=tresh, format_im=type_im)
else:
    raise ValueError('Model name not understood: {0}. Must be in {{Unet3D, Unet2D, LSTM}}.'.format(model_name))

# Display training inforamtions
visualize.display_loss(name,save=True)

# Make video
if VIDEO and not(model_name=='Unet2D'):
    visualize.save_png_video(net, X_test, thresh=tresh, format_im=type_im,name=name+'_demo_video')
    visualize.automatic_process(name+'_demo_video',cut=10,nb_movie=2,nb_seq_in_video=130)

## Compute score on test
t_pred = time.time()
y_pred = np.array(net.predict(X_test)>tresh,np.uint8).flatten()
t_pred = time.time() - t_pred

Y_test = np.array(Y_test>0.5,dtype=np.uint8)
dice_score = losses.dice_coef(Y_test.flatten(), y_pred.flatten())
jaccard_score = losses.jaccard_coef(Y_test.flatten(), y_pred.flatten())
acc = losses.accuracy(Y_test.flatten(), y_pred.flatten())
print("Prediction done in {0} seconds for {1} frames: {2} frame per second.".format(t_pred,X_test.shape[0],t_pred/X_test.shape[0]))
print("Dice score: {0}. Jaccard score: {1}. Accuracy {2}.".format(dice_score,jaccard_score,acc))
