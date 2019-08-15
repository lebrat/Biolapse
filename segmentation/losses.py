from keras import backend as K
import numpy as np

def jaccard_coef(y_true, y_pred):
    intersec = y_true*y_pred
    union = np.logical_or(y_true, y_pred).astype(int)
    if intersec.sum() == 0:
        jac_coef = 0
    else:
        jac_coef = round(intersec.sum()/union.sum(), 2)
    return jac_coef

## dice coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
def dice_coef(y_true, y_pred, smooth=1):
    intersec = y_true*y_pred
    union = y_true+y_pred-intersec
    if intersec.sum() == 0:
        dice_coef = 0
    else:
        dice_coef = round((intersec.sum()*2+smooth)/(union.sum()+smooth), 2)
    return dice_coef

def dice_coef_K(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss_K(y_true, y_pred):
    return 1-dice_coef_K(y_true, y_pred)
