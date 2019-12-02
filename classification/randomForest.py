import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import cv2


ed = 21

im_train = np.load(os.path.join('..','Data','Classification','im_train.npy'))
im_test = np.load(os.path.join('..','Data','Classification','im_test.npy'))
label_train = np.load(os.path.join('..','Data','Classification','feature_train.npy'))
label_test = np.load(os.path.join('..','Data','Classification','feature_test.npy'))



Nfeature = 7
feature_train = np.zeros((label_train.shape[0],Nfeature))

for k in range(im_train.shape[0]):
    # Mean and variance of image
    tmp = im_train[k].copy()
    tmp[im_train[k]==0] = np.nan
    feature_train[k,0] = np.nanmean(tmp)
    feature_train[k,1] = np.nansum((feature_train[k,0]-tmp)**2)/float(np.nansum(tmp>1e-7))

    volume = float(np.nansum(tmp>1e-7))

    # Mean and variance of edges
    mask = np.array(im_train[k]>1e-7,dtype=np.uint8)
    edge = np.array((mask- cv2.erode(mask,np.ones((ed,ed))))>0, dtype=np.float32)
    tmp = edge*im_train[k].copy()
    tmp[im_train[k]==0] = np.nan
    feature_train[k,2] = np.nanmean(tmp)
    feature_train[k,3] = np.nansum((feature_train[k,2]-tmp)**2)/float(np.nansum(tmp>1e-7))

    # Volume/perimeter: should measure some spericity
    perim = np.array((mask- cv2.erode(mask,np.ones((3,3))))>0, dtype=np.float32)
    perim = float(np.sum(perim))
    feature_train[k,4] = volume / perim

    # Aspect ration
    x,y,w,h = cv2.boundingRect(mask)
    aspect_ratio = float(w)/h
    feature_train[k,5] = aspect_ratio

    # Roundness
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (x,y),(MA,ma),angle = cv2.fitEllipse(contours[0])
    eccentricity = np.sqrt((ma/2.)**2-(MA/2.)**2)
    eccentricity = np.round(eccentricity/(x/2),2)
    feature_train[k,6] = eccentricity


nclass=4
clf = RandomForestClassifier(n_estimators=1000, max_depth=10,random_state=0, verbose=1)
clf.fit(feature_train,label_train)


## Prediciton
feature_test = np.zeros((label_test.shape[0],Nfeature))
for k in range(im_test.shape[0]):
    # Mean and variance of image
    tmp = im_test[k].copy()
    tmp[im_test[k]==0] = np.nan
    feature_test[k,0] = np.nanmean(tmp)
    feature_test[k,1] = np.nansum((feature_test[k,0]-tmp)**2)/float(np.nansum(tmp>1e-7))

    volume = float(np.nansum(tmp>1e-7))

    # Mean and variance of edges
    mask = np.array(im_train[k]>1e-7,dtype=np.uint8)
    edge = np.array((mask- cv2.erode(mask,np.ones((ed,ed))))>0, dtype=np.float32)
    tmp = edge*im_train[k].copy()
    tmp[im_train[k]==0] = np.nan
    feature_test[k,2] = np.nanmean(tmp)
    feature_test[k,3] = np.nansum((feature_test[k,2]-tmp)**2)/float(np.nansum(tmp>1e-7))

    # Volume/perimeter: should measure some spericity
    perim = np.array((mask- cv2.erode(mask,np.ones((3,3))))>0, dtype=np.float32)
    perim = float(np.sum(perim))
    feature_test[k,4] = volume / perim

    # Aspect ration
    x,y,w,h = cv2.boundingRect(mask)
    aspect_ratio = float(w)/h
    feature_test[k,5] = aspect_ratio

    # Roundness
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (x,y),(MA,ma),angle = cv2.fitEllipse(contours[0])
    eccentricity = np.sqrt((ma/2)**2-(MA/2)**2)
    eccentricity = np.round(eccentricity/(x/2),2)
    feature_test[k,6] = eccentricity


predict_test = clf.predict(feature_test)

import pylab; pylab.ion()
proba_test_predict = np.argmax(predict_test,1)
proba_test = np.argmax(label_test,1)
pylab.plot(proba_test_predict,'r')
pylab.plot(proba_test,'k')