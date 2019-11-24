import numpy as np
import imageio
from skimage.transform import resize
import glob
import os
from utils import get_labels, stringSplitByNumbers

"""
Merge data extracted by Biolapse interface and save them into one unique folder to train a neural network to predict phases.
"""

def extract_data(path_to_crop,nx,ny,nclass,seq_test=0,save_png=True):
    if not os.path.exists(os.path.join('..','Data','Classification','Train')):
        os.makedirs(os.path.join('..','Data','Classification','Train'))	
    if not os.path.exists(os.path.join('..','Data','Classification','Test')):
        os.makedirs(os.path.join('..','Data','Classification','Test'))	

    im_train_list = []
    im_test_list = []
    feature_train_list = []
    feature_test_list = []

    for fol in os.listdir(os.path.join(path_to_crop)):
        ## Load data
        labels, idx = get_labels(os.path.join(path_to_crop,fol))
        all_img = glob.glob(os.path.join(path_to_crop,fol, 'zoom','*', '*.png'))
        all_img = sorted(all_img,key=stringSplitByNumbers)
        all_masks = glob.glob(os.path.join(path_to_crop,fol, 'zoom_mask', '*', '*.png'))
        all_masks = sorted(all_masks,key=stringSplitByNumbers)
        n_im = len(all_img)

        labels_arr = np.zeros((n_im,nclass))
        idx_arr = np.zeros((n_im,1))
        label_list_arr = np.zeros((n_im,1))
        id_img = []
        nb_seq = len(labels)
        idreverse = np.zeros((n_im,2))
        cpt=0
        for i in range(len(labels)):
            id_img_ = []
            for j in range(len(labels[i])):
                id_img_.append(cpt)
                if labels[i][j] == 4: # G1 = G2
                    labels_arr[cpt,0]=1
                    label_list_arr[cpt] = 0
                else:
                    labels_arr[cpt,labels[i][j]]=1
                    label_list_arr[cpt] = labels[i][j]
                idx_arr[cpt] = idx[i][j]
                idreverse[cpt,0] = i
                idreverse[cpt,1] = j
                cpt+=1
            id_img.append(id_img_)

        nb_test = len(labels[seq_test])
        im_train = np.zeros((n_im-nb_test,nx,ny))
        im_test = np.zeros((nb_test,nx,ny))
        feature_train = np.zeros((n_im-nb_test,nclass))
        feature_test = np.zeros((nb_test,nclass))
        cpt_test = 0
        cpt_train = 0
        for k in range(len(all_img)):
            mask_tmp = np.array(resize(imageio.imread(all_masks[int(idx_arr[k])]),
                            (nx, ny))>1e-10, dtype=np.float32)
            im_ = all_img[int(idx_arr[k])]
            im_tmp = np.array(resize(imageio.imread(im_),
                                (nx, ny)), dtype=np.float32)
            im_tmp = np.array(
                ((im_tmp-im_tmp.min())/(im_tmp.max()-im_tmp.min()))*mask_tmp, dtype=np.float32)
            if (idreverse[k,0]== seq_test):
                im_test[cpt_test] = im_tmp
                feature_test[cpt_test] = labels_arr[k]
                if save_png:
                    im_tmp = np.array((im_tmp-np.max(im_tmp))/(np.max(im_tmp)-np.min(im_tmp)),dtype=np.uint8)
                    imageio.imsave(os.path.join('..','Data','Classification','Test',str(cpt_test).zfill(5)+'.png'),im_tmp)
                cpt_test += 1
            else:
                im_train[cpt_train] = im_tmp
                feature_train[cpt_train] = labels_arr[k]
                if save_png:
                    im_tmp = np.array((im_tmp-np.max(im_tmp))/(np.max(im_tmp)-np.min(im_tmp)),dtype=np.uint8)
                    imageio.imsave(os.path.join('..','Data','Classification','Train',str(cpt_train).zfill(5)+'.png'),im_tmp)
                cpt_train += 1
        im_train_list.append(im_train)
        im_test_list.append(im_test)
        feature_train_list.append(feature_train)
        feature_test_list.append(feature_test)

    im_train_list_ = []
    feature_train_list_ = []
    for k in range(len(im_train_list)):
        for t in range(im_train_list[k].shape[0]):
            im_train_list_.append(im_train_list[k][t])
            feature_train_list_.append(feature_train_list[k][t])
    im_test_list_ = []
    feature_test_list_ = []
    for k in range(len(im_test_list)):
        for t in range(im_test_list[k].shape[0]):
            im_test_list_.append(im_test_list[k][t])
            feature_test_list_.append(feature_test_list[k][t])

    im_train = np.array(im_train_list_)
    im_test = np.array(im_test_list_)
    feature_train = np.array(feature_train_list_)
    feature_test = np.array(feature_test_list_)

    np.save(os.path.join('..','Data','Classification','im_train.npy'),im_train)
    np.save(os.path.join('..','Data','Classification','im_test.npy'),im_test)
    np.save(os.path.join('..','Data','Classification','feature_train.npy'),feature_train)
    np.save(os.path.join('..','Data','Classification','feature_test.npy'),feature_test)


def main():

    ## Parameters
    path_to_crop='Outputs'

    nx = ny = 128
    nclass = 4 # G, erly S, mid S, late S

    seq_test = 0 # number of the test sequence
    save_png = True
    
    extract_data(path_to_crop,nx,ny,nclass,seq_test,save_png)

if __name__ == '__main__':
    main()