from keras import backend as K
import numpy as np

def jaccard_coef(y_true, y_pred):
    intersec = y_true*y_pred
    union = np.logical_or(y_true, y_pred).astype(int)
    if intersec.sum() == 0:
        jac_coef = 0
    else:
        jac_coef = round(intersec.sum()/union.sum(), 2)
    # intersection = np.logical_and(y_true, y_pred)
    # union = np.logical_or(y_true, y_pred)
    # iou_score = np.sum(intersection) / np.sum(union)
    return jac_coef

def dice_coef(y_true, y_pred, smooth=1):
    intersec = np.logical_and(y_true, y_pred).astype(int)
    sum_ = y_true.sum()+y_pred.sum()
    if intersec.sum() == 0:
        dice_coef = 0
    else:
        dice_coef = round((intersec.sum()*2+smooth)/(sum_+smooth), 2)
    return dice_coef

def dice_coef_K(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss_K(y_true, y_pred):
    return 1-dice_coef_K(y_true, y_pred)

def accuracy(y_true,y_pred):
    return np.sum(np.equal(y_true,y_pred.flatten()))/len(y_true)

