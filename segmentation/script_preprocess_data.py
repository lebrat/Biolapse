import numpy as np
import os

import data_loader

"""
This script is a demo of the kind of preprocessing that can be used in this project.
"""

## Parameters
type_im = np.uint8
nx = ny = 256

## Tif to png
# Convert all tif images from a directory and subdirectory into png images.
path_folder_tif = '../Data/Acquisitions/Test/images'
path_to_save = '../Data/Acquisitions/png/Test/images'
data_loader.tif2png(path_folder_tif,path_to_save,type_im,nx,ny)
path_folder_tif = '../Data/Acquisitions/Test/masks'
path_to_save = '../Data/Acquisitions/png/Test/masks'
data_loader.tif2png(path_folder_tif,path_to_save,type_im,nx,ny)
path_folder_tif = '../Data/Acquisitions/Train/images'
path_to_save = '../Data/Acquisitions/png/Train/images'
data_loader.tif2png(path_folder_tif,path_to_save,type_im,nx,ny)
path_folder_tif = '../Data/Acquisitions/Train/masks'
path_to_save = '../Data/Acquisitions/png/Train/masks'
data_loader.tif2png(path_folder_tif,path_to_save,type_im,nx,ny)

# Data augmentation
path_train_save = '../Data/Segmentation/Train3D/augmentation'
path_train_im = '../Data/Acquisitions/png/Train/images/'
path_train_mask = '../Data/Acquisitions/png/Train/masks/'
for dirnames in  os.listdir(path_train_im):
    path_imgs = os.path.join(path_train_im,dirnames,'*.png')
    path_masks = os.path.join(path_train_mask,dirnames,'*.png')
    path_save = os.path.join(path_train_save,dirnames)
    data_loader.augmentation_png(path_imgs,path_masks,path_save, original=True, oneOfTwo=True, reverse=True, reverseOneOfTwo=True,
    rotation1=0, rotation2=180,rotation3=0,rotation4=90,rotation5=270,rotation6=0, rotation7=0)
path_test_save = '../Data/Segmentation/Test3D/augmentation'
path_test_im = '../Data/Acquisitions/png/Test/images/'
path_test_mask = '../Data/Acquisitions/png/Test/masks/'
for dirnames in  os.listdir(path_test_im):
    path_imgs = os.path.join(path_test_im,dirnames,'*.png')
    path_masks = os.path.join(path_test_mask,dirnames,'*.png')
    path_save = os.path.join(path_test_save,dirnames)
    data_loader.augmentation_png(path_imgs,path_masks,path_save, original=True, oneOfTwo=False, reverse=True, reverseOneOfTwo=False,
    rotation1=0, rotation2=180,rotation3=0,rotation4=90,rotation5=0,rotation6=0, rotation7=0)





## If no augmentation requiered
# Data augmentation
path_train_save = '../Data/Segmentation/Train2D/augmentation'
path_train_im = '../Data/Acquisitions/png2D/Train/images/'
path_train_mask = '../Data/Acquisitions/png2D/Train/masks/'
for dirnames in  os.listdir(path_train_im):
    path_imgs = os.path.join(path_train_im,dirnames,'*.png')
    path_masks = os.path.join(path_train_mask,dirnames,'*.png')
    path_save = os.path.join(path_train_save,dirnames)
    data_loader.augmentation_png(path_imgs,path_masks,path_save, original=True, oneOfTwo=False, reverse=False, reverseOneOfTwo=False,
    rotation1=0, rotation2=0,rotation3=0,rotation4=0,rotation5=0,rotation6=0, rotation7=0)
path_test_save = '../Data/Segmentation/Test2D/augmentation'
path_test_im = '../Data/Acquisitions/png2D/Test/images/'
path_test_mask = '../Data/Acquisitions/png2D/Test/masks/'
for dirnames in  os.listdir(path_test_im):
    path_imgs = os.path.join(path_test_im,dirnames,'*.png')
    path_masks = os.path.join(path_test_mask,dirnames,'*.png')
    path_save = os.path.join(path_test_save,dirnames)
    data_loader.augmentation_png(path_imgs,path_masks,path_save, original=True, oneOfTwo=False, reverse=False, reverseOneOfTwo=False,
    rotation1=0, rotation2=0,rotation3=0,rotation4=0,rotation5=0,rotation6=0, rotation7=0)

