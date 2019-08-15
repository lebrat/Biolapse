import sys
from PyQt5.QtCore import Qt ,pyqtSignal, QRect
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QProgressBar
import uuid
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import imageio
import glob
import cv2
import os
import shutil
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
	QVBoxLayout, QWidget, QFileDialog, QPushButton

from tensorflow.python.client import device_lib
from keras.models import load_model
import keras
from keras.losses import binary_crossentropy
import keras.optimizers as optimizers

