import numpy as np
import cv2
import os
from keras.models import load_model
import imageio
from skimage.transform import resize
from classification.utils import find_connected_components
from tracking.tracking import extractMaskFromPoint, draw_border
import torch
import itertools,operator
import shutil
from classification.phase_prediction import phase_prediction


"""
Use segmented time series images to detect cells and track them in time. Then the phase is predicted 
using Viterbi and a neural network trained to predict phase probability.

INPUT: 
    - im: original images.
    - im_channel (optional): second channel images.
    - masks: segmentation of 'im'.
    - max_speed: maximum speed in pixel allow between one cell in one time step.
    - max_volume_var: maximum volume variation (in %) allow between one cell in one time step.
    - minimalSize: minimal size of an admissible cell.
    - nx_crop, ny_crop: size of the crop, should be adequate to the neural network.
    - save_out: if True, save images in png format, in Data/Biolapse.
    - step: number of image between which we try to detect new cells; If to small, processing might be too long.
"""


def im2phase(im_,im_channel,masks,max_speed_disp=20,max_volume_var=0.3,minimalSize=100,nx_crop=128,ny_crop=128,save_out=True,step=20):


    mask_ = np.array(255*(masks>1e-7),dtype=np.uint8)
    mask_final = np.zeros_like(mask_)
    # import ipdb; ipdb.set_trace()
    for i in range(mask_.shape[0]):
        # Erode
        kernel = np.ones((5,5), np.uint8) 
        mask_erosion = cv2.erode(mask_[i], kernel, iterations=1) 
        # Dilate
        kernel = np.ones((4,4), np.uint8) 
        mask_final[i] = cv2.dilate(mask_erosion, kernel, iterations=1) 

    ## Find connected components
    mask_crop_list = []
    im_crop_list = []
    im_crop2_list = []
    im = (im_ - np.min(im_))/(np.max(im_)-np.min(im_))
    im = np.array(im*255,dtype=np.uint8)
    im_final = im.copy()
    mask_final = np.array(255*(mask_final>1e-7),dtype=np.uint8)
    mask_all = np.zeros((im.shape[0],im.shape[1],im.shape[2]),dtype=np.float32)
    # Initialization
    for l in range(0,np.max(mask_final.shape[0]-step,0),step):
        num_,im_num = find_connected_components(mask_final[l],minimalSize)
        # Check no redundasncy
        num = []
        for t in range(len(num_)):
            pos = np.round(num_[t][1])
            if mask_all[l,int(pos[0]),int(pos[1])]==0:
                num.append(num_[t])
        for k in range(len(num)):
            im_out, DictBar, mask_crop, im_crop, im_crop2, maskFinal = extractMaskFromPoint(mask_final, im, im_channel, l, num[k][1], im.shape[0]-1, -1, minimalSize)
            # compute speed of each components
            v = np.ones(len(DictBar)-1)*max_speed_disp*2
            vol = np.ones(len(DictBar)-1)*max_volume_var*2
            for i in range(l+1,l+len(DictBar)):
                tmp  = np.array(mask_crop[i],dtype=np.float32)
                tmp_ = np.array(mask_crop[i-1],dtype=np.float32)
                v[i-1-l] = np.linalg.norm(DictBar[i]-DictBar[i-1])
                vol[i-1-l] = np.abs(np.sum(tmp)-np.sum(tmp_))/(np.sum(tmp))

            # Find longest sequence not exceeding speed
            A = np.array(v>max_speed_disp,dtype=np.float32)
            B = np.array(vol>max_volume_var,dtype=np.float32)
            rA = max((list(y) for (x,y) in itertools.groupby((enumerate(A)),operator.itemgetter(1)) if x == 0), key=len)
            rB = max((list(y) for (x,y) in itertools.groupby((enumerate(B)),operator.itemgetter(1)) if x == 0), key=len)
            r_st = int(np.max((rA[0][0],rB[0][0])))
            r_end = int(np.min((rA[-1][0],rB[-1][0])))
            if rA[0][1]==0 and rB[0][1]==0 and np.sum(mask_all[i]*maskFinal[i])==0 and r_st<r_end: # longest sequence not exceeding speed
                tmp_im =[]
                tmp_mask = []
                tmp_im2 = []
                for i in range(r_st,r_end+1):
                    tmp_mask.append(resize(mask_crop[l+i],(nx_crop,ny_crop)))
                    tmp_im.append(resize(im_crop[l+i]*mask_crop[l+i],(nx_crop,ny_crop)))
                    if len(im_channel)>1:
                        tmp_im2.append(resize(im_crop2[i],(nx_crop,ny_crop)))
                    mask_all[l+i] += np.array(maskFinal[l+i]>1e-5,dtype=np.float32)
                im_crop_list.append(tmp_im)
                mask_crop_list.append(tmp_mask)
                im_crop2_list.append(tmp_im2)


    for i in range(im.shape[0]):
        mask_tmp = np.array(255*(mask_all[i]>1e-5),dtype=np.uint8)
        kernel = np.ones((4,4), np.uint8) 
        mask_tmp = cv2.dilate(mask_tmp, kernel, iterations=1) 
        im_final[i] = np.dot(draw_border(mask_tmp,im[i].copy()), [0.9, 0.587, 0.144])
    if save_out:
        if not os.path.exists(os.path.join('Data','Biolapse','tracking','all')):
            os.makedirs(os.path.join('Data','Biolapse','tracking','all'))
        for i in range(im_final.shape[0]):
            imageio.imsave(os.path.join('Data','Biolapse','tracking','all',str(i).zfill(5)+'.png'),np.squeeze(im_final[i]))
    # np.save(os.path.join('Data','im_crop.npy'),im_crop_list)
    # np.save(os.path.join('Data','mask_crop.npy'),mask_crop_list)
    # np.save(os.path.join('Data','im_crop2.npy'),im_crop2_list)

    ## Apply phase prediction on each track
    # im_crop_list=np.load(os.path.join('Data','im_crop.npy'),allow_pickle=True)
    # mask_crop_list=np.load(os.path.join('Data','mask_crop.npy'),allow_pickle=True)
    # im_crop2_list=np.load(os.path.join('Data','im_crop2.npy'),allow_pickle=True)
    from classification.utils import Net
    model = Net()
    model.load_state_dict(torch.load(os.path.join('Data','Classification','mytosis_cnn.pt')))
    # model = torch.load(os.path.join('Data','Classification','mytosis_cnn.pt'))
    model.eval()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)

    phase_pred = []
    for k in range(len(im_crop_list)):
        phase_pred_, _ = phase_prediction(model,np.array(im_crop_list[k]),np.zeros(1),np.zeros(1),False,n_max=40,nclass=4,p_tr=0.05*np.ones(4))
        phase_pred.append(phase_pred_)
    return phase_pred, im_crop_list, im_crop2_list, im_final