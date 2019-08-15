import numpy as np
import pylab
import visualize
import data_loader
from keras.models import load_model
import losses

## Parameters
# name = 'nn_Aude_Unet3D'
name = 'nn_Aude_Unet3D_256_32best'
batch_size = 2
type_im = np.uint16
type_tf = np.float32
nx = ny = 256
TIME = 10
LSTM = False
## Data loader 
path_test = 'AudeGuenole/Test/SUM_PCNA_74_3Dguassianblur/Original'
# path_test = '/media/valentin/Data/Documents/Projets/ML/mldenoise/Tracking/Data/CellChallenge/PNG/test/augmentation/Original'


# Unet 3D
X_test, Y_test= data_loader.path_to_time_batchs(path_test,nx,ny,TIME=TIME,type_im=type_im,format_im='png',
	code_im1='images',code_im2='masks')
X_test = np.expand_dims(np.array(X_test, dtype=np.float32)/np.iinfo(type_im).max,4)
Y_test = np.expand_dims(np.array(Y_test, dtype=np.float32)/np.iinfo(type_im).max,4)
if LSTM:
    X_test = np.rollaxis(X_test,3,1)
    Y_test = np.rollaxis(Y_test,3,1)

from keras.losses import binary_crossentropy
l1 = 1.
l2 = -1.
loss1 = binary_crossentropy
loss2 = losses.dice_coef_loss_K
def custom_loss(y_true,y_pred):
    return l1*loss1(y_true,y_pred)+l2*loss2(y_true,y_pred)
path_h5 = 'Data/Model/'+name+'.h5'
path_h5 = 'Data/Weights/'+name+'.hdf5'
net = load_model(path_h5,custom_objects={'custom_loss':custom_loss})
# net = load_model(path_h5)
net.summary()

tresh = 0.5
## Save prediction
if LSTM:
    y_pred = visualize.save_png_list_channel_first(net, X_test, name=name, tresh=tresh, format_im=type_im) # Save in Outputs
else:
    y_pred = visualize.save_png_list(net, X_test, name=name, tresh=tresh, format_im=type_im) # Save in Outputs

visualize.display_images_temps(net,X_test,N=1,tresh=tresh)

# Save loss
visualize.display_loss(name,save=True)

## Compute score on test
# y_pred = net.predict(X_test)

dice_score = losses.dice_coef(Y_test.flatten(), y_pred.flatten())
jaccard_score = losses.jaccard_coef(Y_test.flatten(), y_pred.flatten())
print("Dice score: {0}. Jaccard score: {1}.".format(dice_score,jaccard_score))


