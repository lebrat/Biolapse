import numpy as np
import os

import data_loader

"""
This script is a demo of the kind of preprocessing that can be used in this project.
"""

## Parameters
type_im = np.uint16
nx = ny = 256
TIME = 10

path_train = 'Data/Train'
path_test = 'Data/Test'

## Tif to png
# Convert all tif images from a directory and subdirectory into png images.
path_folder_tif = './Data/tif'
path_to_save = './Data/tif/png_conversion'
data_loader.tif2png(path_folder_tif,path_to_save,type_im,nx,ny)

# Data augmentation
path_imgs = path_masks ='./Data/tif/png_conversion/SUM_PCNA_74_3Dguassianblur/*.png'
path_save = './Data/tif/augmentation'
data_loader.augmentation_png(path_imgs,path_masks,path_save, original=True, oneOfTwo=True, reverse=True, reverseOneOfTwo=True,
 rotation1=30, rotation2=150,rotation3=240,rotation4=90,rotation5=290,rotation6=120, rotation7=330)

