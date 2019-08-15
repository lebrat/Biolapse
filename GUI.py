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

"""
Graphical interface - allow experienced user to label cycle of cells. 

All outputs are saved into a folder 'Outputs' within the folder containing the images.
The saved files are:
	- cells.csv: a csv file containing the time at which start eache phase. It also contains 
	the barycenter of the mask associated to the considered cell.
	- Outputs/masks: the masks are saved in this folder when computed whith neural network. If masks
	 are loaded, they ae not saved.
	- Outputs/zoom: save crop of images analysed with the interface. If a second channel is given, 
	 they ar saved in the same folder with extension 'channel2_' preceding the name.

Authors: Valentin Debarnot, Léo Lebrat. 17/07/2019.
"""

# TODO: save image in 16 bits if given in 16 bits.

## Uncomment to force not use GPU.
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

ratioKept = True # Keep ratio of the images display.
type_im = np.uint16 # Format wanted by the neural network.
minimalSize = 70 # Minimal number of pixel to say a mask is valid.
LSTM = False # Use a LSTM network.


"""
INPUT
	- masks: time serie of masks associated to im. Type must be np.uint8.
	- im: original time images. Type must be np.uint8.
	- imageStart: number of the first image into which display the mask.
	- pos: center of the cell which we want to compute the mask.
	- finish: number of the last frame into which display the mask.
	- progressbar:
	- alpha (default = 0.25):
OUTPUT
	- Time serie images containing original image and border of the masks close to 'pos'.
"""
# TODO: utiliser un tracking plus fin (dilatations, etc..).
def extractMaskFromPoint(masks,im,im_channel,imageStart,pos,finish,progressbar,alpha=.25):
	if len(im_channel.shape)==1:
		secondChannel = False
	else:
		secondChannel = True
	barycenter = np.copy(pos)
	nx_mask = masks.shape[1]
	ny_mask = masks.shape[2]
	barycenter[0] = nx_mask - barycenter[0]
	DictBar = {} # contain barycenter and image index 
	DictBar[imageStart] = barycenter.copy()
	im_focus = {}
	im_channel_focus = {}

	# Detect border
	im_out = np.repeat(np.expand_dims(im.copy(),3),3,3) # Need to turn into rgb image
	binary = cv2.threshold(masks[imageStart], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	num,compo=cv2.connectedComponents(binary)
	localCells = []
	for chunckNo in range(1,num):
		if len(np.where(compo==chunckNo)[0]) < minimalSize:
			compo[np.where(compo==chunckNo)] = 0
		else:
			localCells += [[chunckNo,(np.where(compo==chunckNo)[0].mean(),np.where(compo==chunckNo)[1].mean()),len(np.where(compo==chunckNo)[0])]]

	# Compute a cost function associated to each large enough mask and keep track of the minimum value
	minCost = np.inf
	indexMin = -1		
	if len(localCells)==0:
		print('No mask found.')
	for ind,elmt in enumerate(localCells):
		cost = np.linalg.norm(elmt[1] - barycenter)
		if cost < minCost:
			indexMin = ind
			minCost = cost
	maskFinal = np.zeros_like(masks[imageStart])
	maskFinal[np.where(compo==localCells[indexMin][0])] = 1 
	
	# select image focus
	delta = 2
	r = np.where(maskFinal)
	max_x = np.min((np.max(r[0])+delta,im.shape[1]))
	max_y = np.min((np.max(r[1])+delta,im.shape[2]))
	min_x = np.max((np.min(r[0])-delta,0))
	min_y = np.max((np.min(r[1])-delta,0))
	im_focus[imageStart] = im[imageStart,min_x:max_x,min_y:max_y]
	if secondChannel:
		im_channel_focus[imageStart] = im_channel[imageStart,min_x:max_x,min_y:max_y]

	print(im.shape,im.max(),im.min())
	# Draw contour
	contours, hierarchy = cv2.findContours(maskFinal.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	im_out[imageStart] = cv2.drawContours(cv2.cvtColor(im[imageStart].astype(np.uint8),cv2.COLOR_GRAY2RGB), contours, -1, (255,0,0), 1)

	# Do it for each time step
	progressbar.setValue(imageStart)
	while imageStart < finish :
		imageStart += 1
		barycenter = np.array(localCells[indexMin][1]) # restart from barycenter of previous mask.
		DictBar[imageStart] = barycenter.copy()

		# Compute cost fucntion
		binary = cv2.threshold(masks[imageStart], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		num,compo=cv2.connectedComponents(binary)
		localCells = []
		for chunckNo in range(1,num):
			if len(np.where(compo==chunckNo)[0]) < minimalSize:
				compo[np.where(compo==chunckNo)] = 0
			else:
				localCells += [[chunckNo,(np.where(compo==chunckNo)[0].mean(),np.where(compo==chunckNo)[1].mean()),len(np.where(compo==chunckNo)[0])]]
		minCost = np.inf
		indexMin = -1
		if len(localCells)==0:
			print('No mask found.')
			break
		for ind,elmt in enumerate(localCells):
			cost = np.linalg.norm(elmt[1] - barycenter)
			if cost < minCost:
				indexMin = ind
				minCost = cost
		maskFinal =np.zeros_like(masks[imageStart])
		maskFinal[np.where(compo==localCells[indexMin][0])] = 1

		# selct image focus
		delta = 2
		r = np.where(maskFinal)
		max_x = np.min((np.max(r[0])+delta,im.shape[1]))
		max_y = np.min((np.max(r[1])+delta,im.shape[2]))
		min_x = np.max((np.min(r[0])-delta,0))
		min_y = np.max((np.min(r[1])-delta,0))
		im_focus[imageStart] = im[imageStart,min_x:max_x,min_y:max_y]
		if secondChannel:
			im_channel_focus[imageStart] = im_channel[imageStart,min_x:max_x,min_y:max_y]
		
		# Draw contour
		contours, hierarchy = cv2.findContours(maskFinal.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		im_out[imageStart] = cv2.drawContours(cv2.cvtColor(im[imageStart].astype(np.uint8),cv2.COLOR_GRAY2RGB), contours, -1, (255,0,0), 1)
		progressbar.setValue(imageStart)
	return im_out.astype(np.uint8), DictBar, im_focus, im_channel_focus


"""
Slider class.
Implement cursor slider and '+' and '-' buttons. Integer value
INPUT
	- minimum: minimum value reachable by cursor.
	- maximum: maximum value reachable by cursor.
	- parent: ?
OUTPUT
	QWidget that can take integer values between 'minimum' and 'maximum'
"""
class Slider(QWidget):
	valueChangedX = pyqtSignal([int], ['QString'])
	def __init__(self, minimum, maximum, parent=None):
		super(Slider, self).__init__(parent=parent)
		self.verticalLayout = QVBoxLayout(self)
		self.label = QLabel(self)
		sample_palette = QPalette()
		sample_palette.setColor(QPalette.WindowText, Qt.white)
		self.label.setPalette(sample_palette)
		self.verticalLayout.addWidget(self.label)
		self.horizontalLayout = QHBoxLayout()
		spacerItem = QSpacerItem(0, 5, QSizePolicy.Expanding, QSizePolicy.Minimum)
		self.horizontalLayout.addItem(spacerItem)
		self.slider = QSlider(self)
		self.slider.setOrientation(Qt.Horizontal)
		self.horizontalLayout.addWidget(self.slider)
		
		# + button
		self.butPlus = QPushButton("+")
		self.butPlus.clicked.connect(self.appui_bouton_plus)
		self.horizontalLayout.addWidget(self.butPlus)

		#- button
		self.butMinus = QPushButton("-")
		self.butMinus.clicked.connect(self.appui_bouton_minus)
		self.horizontalLayout.addWidget(self.butMinus)
		spacerItem3 = QSpacerItem(0, 5, QSizePolicy.Expanding, QSizePolicy.Minimum)
		self.horizontalLayout.addItem(spacerItem3)

		self.verticalLayout.addLayout(self.horizontalLayout)
		self.resize(self.sizeHint())
		self.minimum = minimum
		self.maximum = maximum
		self.slider.valueChanged.connect(self.setLabelValue)
		self.x = None
		self.setLabelValue(self.slider.value())

	def setLabelValue(self, value):
		self.x = self.minimum + int((float(value) / (self.slider.maximum() - self.slider.minimum())) * (
		self.maximum - self.minimum))
		self.valueChangedX.emit(self.x)
		self.label.setText("{0:.4g}".format(self.x))

	def appui_bouton_plus(self):
		if self.x < self.maximum:
			self.x += 1
			self.valueChangedX.emit(self.x)
		self.label.setText("{0:.4g}".format(self.x))

	def appui_bouton_minus(self):
		if self.x > self.minimum:
			self.x -= 1
			self.valueChangedX.emit(self.x)
		self.label.setText("{0:.4g}".format(self.x))


"""
Slider class.
Implement cursor slider. Float value.
INPUT
	- minimum: minimum value reachable by cursor.
	- maximum: maximum value reachable by cursor.
	- parent: ?
OUTPUT
	QWidget that can take float values between 'minimum' and 'maximum'
"""
class Slider_thresh(QWidget):
	valueChangedX = pyqtSignal([float], ['QString'])
	def __init__(self, minimum, maximum, parent=None):
		super(Slider_thresh, self).__init__(parent=parent)
		self.verticalLayout = QVBoxLayout(self)
		self.label = QLabel(self)
		sample_palette = QPalette()
		sample_palette.setColor(QPalette.WindowText, Qt.white)
		self.label.setPalette(sample_palette)
		self.verticalLayout.addWidget(self.label)
		self.horizontalLayout = QHBoxLayout()
		spacerItem = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
		self.horizontalLayout.addItem(spacerItem)
		self.slider = QSlider(self)
		self.slider.setOrientation(Qt.Horizontal)
		self.horizontalLayout.addWidget(self.slider)
		spacerItem1 = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
		self.horizontalLayout.addItem(spacerItem1)
		self.verticalLayout.addLayout(self.horizontalLayout)
		self.resize(self.sizeHint())
		self.minimum = minimum
		self.maximum = maximum
		self.slider.valueChanged.connect(self.setLabelValue)
		self.x = None
		self.setLabelValue(0.5*(self.slider.maximum() - self.slider.minimum()))

		# + button
		self.butPlus = QPushButton("+")
		self.butPlus.clicked.connect(self.appui_bouton_plus)
		self.horizontalLayout.addWidget(self.butPlus)
		#- button
		self.butMinus = QPushButton("-")
		self.butMinus.clicked.connect(self.appui_bouton_minus)
		self.horizontalLayout.addWidget(self.butMinus)
		spacerItem3 = QSpacerItem(0, 5, QSizePolicy.Expanding, QSizePolicy.Minimum)
		self.horizontalLayout.addItem(spacerItem3)

	def setLabelValue(self, value):
		self.x = self.minimum + ((float(value) / (self.slider.maximum() - self.slider.minimum())) * (self.maximum - self.minimum))
		self.valueChangedX.emit(self.x)
		self.label.setText("thresh: {0:.4g}".format(self.x))
	def appui_bouton_plus(self):
		if self.x < self.maximum:
			self.x += 0.1
			self.valueChangedX.emit(self.x)
		self.label.setText("{0:.4g}".format(self.x))
	def appui_bouton_minus(self):
		if self.x > self.minimum:
			self.x -= 0.1
			self.valueChangedX.emit(self.x)
		self.label.setText("{0:.4g}".format(self.x))

"""
Implement buttons to manage interface.
The buttons made are:
	- Quit: quit the application.
	- Channel: load other image (other chanel), and save croped images in all channels.
"""
class InterfaceManagerButton(QWidget):
	isQuit  = pyqtSignal()
	isChannel  = pyqtSignal()
	def __init__(self, parent=None):
		super(InterfaceManagerButton, self).__init__(parent=parent)
		self.verticalLayout = QVBoxLayout(self)
		self.horizontalLayout = QHBoxLayout()

		self.butQuit = QPushButton("Quit")
		self.butQuit.clicked.connect(self.quitImage)
		self.horizontalLayout.addWidget(self.butQuit)

		self.butChannel = QPushButton("Add channel")
		self.butChannel.clicked.connect(self.channelImage)
		self.horizontalLayout.addWidget(self.butChannel)

		spacerItem = QSpacerItem(0, 0.5, QSizePolicy.Expanding, QSizePolicy.Minimum)
		self.horizontalLayout.addItem(spacerItem)
		self.verticalLayout.addLayout(self.horizontalLayout)
		self.resize(self.sizeHint())
	
	## click actions emitting signals
	def quitImage(self):
		print("Quit")
		self.isQuit.emit()
	def channelImage(self):
		print("Adding new channel")
		self.isChannel.emit()

"""
Implement bottom buttons.
The buttons made are:
	- Shoot: use to start shooting procedure, i.e. following barycenter of mask into the ROI defined. 
	- Unshoot: refresh ROI selection.
	- G1, eraly S, mid S, late S, G2, end tracking: use to defined phases of interest of the cell follow
	by the shooting procedure.
"""
class BottomBut(QWidget):
	isShooted   = pyqtSignal()
	isunShooted = pyqtSignal()
	isG1        = pyqtSignal()
	isEarlyS    = pyqtSignal()
	isMidS      = pyqtSignal()
	isLateS     = pyqtSignal()
	isG2        = pyqtSignal()
	isEnd       = pyqtSignal()
	isloadMasks = pyqtSignal()
	iscomputeMasks  = pyqtSignal()
	def __init__(self, parent=None):
		super(BottomBut, self).__init__(parent=parent)
		self.verticalLayout = QVBoxLayout(self)
		self.horizontalLayout = QHBoxLayout()

		self.butShoot = QPushButton("᪠ Shoot")
		self.butShoot.clicked.connect(self.shoot)
		self.horizontalLayout.addWidget(self.butShoot)

		self.butunShoot = QPushButton("᳁ Un-Shoot")
		self.butunShoot.clicked.connect(self.unshoot)
		self.horizontalLayout.addWidget(self.butunShoot)

		spacerItem = QSpacerItem(0, 0.5, QSizePolicy.Expanding, QSizePolicy.Minimum)
		self.horizontalLayout.addItem(spacerItem)

		self.butloadMasks = QPushButton("Load masks")
		self.butloadMasks.clicked.connect(self.loadMasks)
		self.horizontalLayout.addWidget(self.butloadMasks)

		self.butcomputeMasks = QPushButton("Compute masks")
		self.butcomputeMasks.clicked.connect(self.computeMasks)
		self.horizontalLayout.addWidget(self.butcomputeMasks)

		spacerItem = QSpacerItem(0, 1, QSizePolicy.Expanding, QSizePolicy.Minimum)
		self.horizontalLayout.addItem(spacerItem)

		self.butG1 = QPushButton("ᛰ G1")
		self.butG1.clicked.connect(self.G1)
		self.horizontalLayout.addWidget(self.butG1)

		self.butEarlyS = QPushButton("ᛰ early S")
		self.butEarlyS.clicked.connect(self.earlyS)
		self.horizontalLayout.addWidget(self.butEarlyS)

		self.butMidS = QPushButton("ᛰ mid S")
		self.butMidS.clicked.connect(self.midS)
		self.horizontalLayout.addWidget(self.butMidS)

		self.butLateS = QPushButton("ᛰ late S")
		self.butLateS.clicked.connect(self.lateS)
		self.horizontalLayout.addWidget(self.butLateS)

		self.butG2 = QPushButton("ᛰ G2")
		self.butG2.clicked.connect(self.G2)
		self.horizontalLayout.addWidget(self.butG2)

		self.butEnd = QPushButton("ᛰ end Tracking")
		self.butEnd.clicked.connect(self.ending)
		self.horizontalLayout.addWidget(self.butEnd)

		self.verticalLayout.addLayout(self.horizontalLayout)
		self.resize(self.sizeHint())

	## click actions emitting signals
	def shoot(self):
		print("shoot")
		self.isShooted.emit()
	def unshoot(self):
		print("unshoot")
		self.isunShooted.emit()
	def G1(self):
		print("G1")
		self.isG1.emit()
	def earlyS(self):
		print("Early S")
		self.isEarlyS.emit()
	def midS(self):
		print("mid S")
		self.isMidS.emit()
	def lateS(self):
		print("Late S")
		self.isLateS.emit()
	def G2(self):
		print("G2")
		self.isG2.emit()
	def ending(self):
		print('end')
		self.isEnd.emit()
	def loadMasks(self):
		print("shoot")
		self.isloadMasks.emit()
	def computeMasks(self):
		print("shoot")
		self.iscomputeMasks.emit()


class Widget(QWidget):
	def __init__(self, parent=None):
		self.Shooted = False
		super(Widget, self).__init__(parent=parent)
		self.plot_mask = False
		self.secondChannel = False

		# Image selection
		self.fold = QFileDialog.getOpenFileNames(self, "Select files containing images (tiff or png).",
		os.getcwd(),"*.png *.tiff *.tif")
		dir_file = self.fold[0][0][:self.fold[0][0].rfind(os.sep)]

		# Check size of the data and format
		if len(self.fold[0])!=0:
			if  self.fold[0][0].endswith('tiff') or self.fold[0][0].endswith('tif'):
				TIFF = True
			elif not(self.fold[0][0].endswith('png')):
				TIFF = False
				raise ValueError("Format not supported.")
			else:
				TIFF = False
		else:
			TIFF = False
			raise ValueError("Error - no file found.")

		# load images in shape (TIME,nx,ny)
		if TIFF:
			imgs = self.fold[0][0]
			print("Warning: only one tiff can be process.")
			tmp = np.array(imageio.mimread(imgs,memtest=False))
			from scipy.misc import bytescale
			tmp = bytescale(tmp)
			self.im = np.squeeze(np.array(tmp,dtype=np.uint8))
			self.im_nn = np.squeeze(np.array(tmp,dtype=type_im))
			self.start = 0
			self.finish = self.im.shape[0]-1
			path_fold = self.fold[0][0]
		else:
			from skimage import color
			pictList = self.fold[0]
			path_fold = self.fold[0]
			tmp = np.array(imageio.imread(pictList[0]))
			from scipy.misc import bytescale
			tmp = bytescale(tmp)
			self.im = np.array(tmp,dtype=np.uint8)
			self.im = np.zeros((len(pictList),self.im.shape[0],self.im.shape[1]))
			self.im_nn = np.zeros((len(pictList),self.im.shape[1],self.im.shape[2]),dtype=type_im)
			for i in range(len(pictList)):
				tmp = color.rgb2gray(imageio.imread(pictList[i]))
				self.im_nn[i] = np.array(tmp,dtype=type_im)
				tmp = bytescale(tmp)
				self.im[i] = np.array(tmp,dtype=np.uint8)
			self.start = 0
			self.finish = len(pictList)-1
		self.im_original = self.im.copy()
		self.nx = self.im[0].shape[0]
		self.ny = self.im[0].shape[1]
		self.data = self.im[0]
		self.maskLoaded = False

		# Set up environnement
		self.Shooted = False
		pal = QPalette()
		pal.setColor(QPalette.Background, Qt.black)
		self.setAutoFillBackground(True)
		self.setPalette(pal)
		self.horizontalLayout = QVBoxLayout(self)
		self.w1 = Slider(self.start,self.finish)
		self.horizontalLayout.addWidget(self.w1)
		self.win = pg.GraphicsWindow(title="Basic plotting examples")
		p1 = self.win.addPlot()
		self.horizontalLayout.addWidget(self.win)
		self.w2 = BottomBut()
		self.horizontalLayout.addWidget(self.w2)
		self.progress = QProgressBar(self)
		self.progress.setGeometry(0, 0, 300, 25)
		self.progress.setMaximum(self.finish)
		self.horizontalLayout.addWidget(self.progress)
		
		def quitProcedure():
			sys.exit()
		"""
		Load second channel from tif or png images. 
		If shape is not eqaul to the original image, and error is raised and the second channel is not loaded.
		"""
		def newChannelProcedure():
			# Image selection
			fold_ = QFileDialog.getOpenFileNames(self, "Select files containing other channel (tiff or png).",
			os.getcwd(),"*.png *.tiff *.tif")
			# Check size of the data and format
			if len(fold_[0])!=0:
				if  fold_[0][0].endswith('tiff') or fold_[0][0].endswith('tif'):
					TIFF = True
				elif not(fold_[0][0].endswith('png')):
					TIFF = False
					raise ValueError("Format not supported.")
				else:
					TIFF = False
			else:
				TIFF = False
				raise ValueError("Error - no file found.")
			if TIFF:
				imgs = fold_[0][0]
				print("Warning: only one tiff can be process.")
				tmp = np.array(imageio.mimread(imgs,memtest=False))
				from scipy.misc import bytescale
				tmp = bytescale(tmp)
				im_channel = np.squeeze(np.array(tmp,dtype=np.uint8))
			else:
				from skimage import color
				pictList = fold_[0]
				# path_fold = fold_[0]
				tmp = np.array(imageio.imread(pictList[0]))
				from scipy.misc import bytescale
				tmp = bytescale(tmp)
				im_channel = np.array(tmp,dtype=np.uint8)
				im_channel = np.zeros((len(pictList),im_channel.shape[0],im_channel.shape[1]))
				for i in range(len(pictList)):
					tmp = bytescale(tmp)
					im_channel[i] = np.array(tmp,dtype=np.uint8)
			self.im_channel = im_channel.copy()
			if self.im_channel.shape == self.im.shape:
				self.secondChannel = True
			else:
				raise ValueError('Channel not with same shape as original images.')
			
		def shootingProcedure():
			if self.plot_mask and not(self.Shooted):
				self.Shooted = True
				self.shootID = uuid.uuid4().hex
				self.frameStart = self.w1.x
				initBB = np.array(self.boundingBox)
				middle = np.array(np.mean(initBB,axis=0),dtype=int)
				tmp = middle[0]
				middle[0] = middle[1]
				middle[1] = tmp
				self.progress.show()
				if self.secondChannel:
					self.im, self.currentBar, self.im_focus, self.im_channel_focus = extractMaskFromPoint(self.masks,self.im,self.im_channel,self.frameStart,middle,self.finish,self.progress,alpha=.25)
				else:
					self.im, self.currentBar, self.im_focus, _ = extractMaskFromPoint(self.masks,self.im,np.zeros(1),self.frameStart,middle,self.finish,self.progress,alpha=.25)
				updateImage()
			elif not(self.plot_mask):
				print('Error: load or compute mask first.')
				return 0
			elif self.Shooted:
				print('Error: already in shooting procedure.')
				return 0
		def unShoot():
			if self.plot_mask and self.Shooted:
				self.Shooted = False
				self.frameStart = None
				self.currentBar = None
				self.shootID = None
				self.im = self.im_original.copy()
				updateImage()
			elif not(self.plot_mask):
				print('Error: load or compute mask first.')
				return 0
			elif not(self.Shooted):
				print('Error: already in unshooting procedure.')
				return 0

		"""
		Writting into csv file.
		The columns of the file are as follows:
			- Code of the current shooting procedure, unique.
			- Time where G1 begin.
			- Position of barycenter of mask when G1 begin.
			- Time where early S begin.
			- Position of barycenter of mask when early S begin.
			- Time where mid S begin.
			- Position of barycenter of mask when mid S begin.
			- Time where late S begin.
			- Position of barycenter of mask when late S begin.
			- Time where G2 begin.
			- Position of barycenter of mask when G2 begin.
			- Time where tracking end.
			- Position of barycenter of mask when tracking ended.
		"""
		if not os.path.exists(os.path.join(dir_file,'Outputs')):
			os.mkdir(os.path.join(dir_file,'Outputs'))
		def G1():
			self.w2.butG1.setStyleSheet("background-color: red")
			if not os.path.exists(os.path.join(dir_file,'Outputs','cells.csv')):
				f= open(os.path.join(dir_file,'Outputs','cells.csv'),"w+")
			else:
				f= open(os.path.join(dir_file,'Outputs','cells.csv'),"a+")
			if self.Shooted:
				f.write(str(self.shootID)+","+str(self.w1.x)+","+str(self.currentBar[self.w1.x])+", , , , , , , , , ,\n")				
				self.prev_shoot = self.w1.x
			else:
				print('Not in shooting mode.')
			f.close()
		def earlyS():
			self.w2.butG1.setStyleSheet("background-color: white")
			self.w2.butEarlyS.setStyleSheet("background-color: red")
			if not os.path.exists(os.path.join(dir_file,'Outputs','cells.csv')):
				f= open(os.path.join(dir_file,'Outputs','cells.csv'),"w+")
			else:
				f= open(os.path.join(dir_file,'Outputs','cells.csv'),"a+")
			if self.Shooted:
				f.write(str(self.shootID)+", , ,"+str(self.w1.x)+","+str(self.currentBar[self.w1.x])+", , , , , , , ,\n")
				# Save masks in same folder than images			
				if not os.path.exists(os.path.join(dir_file,'Outputs','zoom',str(self.shootID))):
					os.makedirs(os.path.join(dir_file,'Outputs','zoom',str(self.shootID)))	
				for k in range(self.prev_shoot,self.w1.x+1):
					tmp = np.array(self.im_focus[k],dtype=np.float32)
					tmp = np.array(255*(tmp-tmp.min())/(tmp.max()-tmp.min()),dtype=np.uint8)
					imageio.imsave(os.path.join(dir_file,'Outputs','zoom',str(self.shootID),str(k).zfill(10)+'.png'),tmp)
					tmp = np.array(self.im_channel_focus[k],dtype=np.float32)
					tmp = np.array(255*(tmp-tmp.min())/(tmp.max()-tmp.min()),dtype=np.uint8)
					imageio.imsave(os.path.join(dir_file,'Outputs','zoom',str(self.shootID),'channel2_'+str(k).zfill(10)+'.png'),tmp)
				self.prev_shoot = self.w1.x
			else:
				print('Not in shooting mode.')
			f.close()
		def midS():
			self.w2.butEarlyS.setStyleSheet("background-color: white")
			self.w2.butMidS.setStyleSheet("background-color: red")
			if not os.path.exists(os.path.join(dir_file,'Outputs','cells.csv')):
				f= open(os.path.join(dir_file,'Outputs','cells.csv'),"w+")
			else:
				f= open(os.path.join(dir_file,'Outputs','cells.csv'),"a+")
			if self.Shooted:
				f.write(str(self.shootID)+", , , , ,"+str(self.w1.x)+","+str(self.currentBar[self.w1.x])+", , , , , ,\n")
				# Save masks in same folder than images			
				if not os.path.exists(os.path.join(dir_file,'Outputs','zoom',str(self.shootID))):
					os.makedirs(os.path.join(dir_file,'Outputs','zoom',str(self.shootID)))	
				for k in range(self.prev_shoot+1,self.w1.x+1):
					tmp = np.array(self.im_focus[k],dtype=np.float32)
					tmp = np.array(255*(tmp-tmp.min())/(tmp.max()-tmp.min()),dtype=np.uint8)
					imageio.imsave(os.path.join(dir_file,'Outputs','zoom',str(self.shootID),str(k).zfill(10)+'.png'),tmp)
					tmp = np.array(self.im_channel_focus[k],dtype=np.float32)
					tmp = np.array(255*(tmp-tmp.min())/(tmp.max()-tmp.min()),dtype=np.uint8)
					imageio.imsave(os.path.join(dir_file,'Outputs','zoom',str(self.shootID),'channel2_'+str(k).zfill(10)+'.png'),tmp)
				self.prev_shoot = self.w1.x
			else:
				print('Not in shooting mode.')
			f.close()
		def lateS():
			self.w2.butMidS.setStyleSheet("background-color: white")
			self.w2.butLateS.setStyleSheet("background-color: red")
			if not os.path.exists(os.path.join(dir_file,'Outputs','cells.csv')):
				f= open(os.path.join(dir_file,'Outputs','cells.csv'),"w+")
			else:
				f= open(os.path.join(dir_file,'Outputs','cells.csv'),"a+")
			if self.Shooted:
				f.write(str(self.shootID)+", , , , , , ,"+str(self.w1.x)+","+str(self.currentBar[self.w1.x])+", , , ,\n")
				# Save masks in same folder than images			
				if not os.path.exists(os.path.join(dir_file,'Outputs','zoom',str(self.shootID))):
					os.makedirs(os.path.join(dir_file,'Outputs','zoom',str(self.shootID)))	
				for k in range(self.prev_shoot+1,self.w1.x+1):
					tmp = np.array(self.im_focus[k],dtype=np.float32)
					tmp = np.array(255*(tmp-tmp.min())/(tmp.max()-tmp.min()),dtype=np.uint8)
					imageio.imsave(os.path.join(dir_file,'Outputs','zoom',str(self.shootID),str(k).zfill(10)+'.png'),tmp)
					tmp = np.array(self.im_channel_focus[k],dtype=np.float32)
					tmp = np.array(255*(tmp-tmp.min())/(tmp.max()-tmp.min()),dtype=np.uint8)
					imageio.imsave(os.path.join(dir_file,'Outputs','zoom',str(self.shootID),'channel2_'+str(k).zfill(10)+'.png'),tmp)
				self.prev_shoot = self.w1.x
			else:
				print('Not in shooting mode.')

			f.close()
		def G2():
			self.w2.butLateS.setStyleSheet("background-color: white")
			self.w2.butG2.setStyleSheet("background-color: red")
			if not os.path.exists(os.path.join(dir_file,'Outputs','cells.csv')):
				f= open(os.path.join(dir_file,'Outputs','cells.csv'),"w+")
			else:
				f= open(os.path.join(dir_file,'Outputs','cells.csv'),"a+")
			if self.Shooted:
				f.write(str(self.shootID)+", , , , , , , , ,"+str(self.w1.x)+","+str(self.currentBar[self.w1.x])+", ,\n")
				# Save masks in same folder than images			
				if not os.path.exists(os.path.join(dir_file,'Outputs','zoom',str(self.shootID))):
					os.makedirs(os.path.join(dir_file,'Outputs','zoom',str(self.shootID)))	
				for k in range(self.prev_shoot+1,self.w1.x+1):
					tmp = np.array(self.im_focus[k],dtype=np.float32)
					tmp = np.array(255*(tmp-tmp.min())/(tmp.max()-tmp.min()),dtype=np.uint8)
					imageio.imsave(os.path.join(dir_file,'Outputs','zoom',str(self.shootID),str(k).zfill(10)+'.png'),tmp)
					tmp = np.array(self.im_channel_focus[k],dtype=np.float32)
					tmp = np.array(255*(tmp-tmp.min())/(tmp.max()-tmp.min()),dtype=np.uint8)
					imageio.imsave(os.path.join(dir_file,'Outputs','zoom',str(self.shootID),'channel2_'+str(k).zfill(10)+'.png'),tmp)
				self.prev_shoot = self.w1.x
			else:
				print('Not in shooting mode.')
			f.close()
		def end():
			self.w2.butG2.setStyleSheet("background-color: white")
			if not os.path.exists(os.path.join(dir_file,'Outputs','cells.csv')):
				f= open(os.path.join(dir_file,'Outputs','cells.csv'),"w+")
			else:
				f= open(os.path.join(dir_file,'Outputs','cells.csv'),"a+")
			if self.Shooted:
				f.write(str(self.shootID)+", , , , , , , , , , ,"+str(self.w1.x)+","+str(self.currentBar[self.w1.x])+"\n")
				# Save masks in same folder than images			
				if not os.path.exists(os.path.join(dir_file,'Outputs','zoom',str(self.shootID))):
					os.makedirs(os.path.join(dir_file,'Outputs','zoom',str(self.shootID)))	
				for k in range(self.prev_shoot+1,self.w1.x+1):
					tmp = np.array(self.im_focus[k],dtype=np.float32)
					tmp = np.array(255*(tmp-tmp.min())/(tmp.max()-tmp.min()),dtype=np.uint8)
					imageio.imsave(os.path.join(dir_file,'Outputs','zoom',str(self.shootID),str(k).zfill(10)+'.png'),tmp)
					tmp = np.array(self.im_channel_focus[k],dtype=np.float32)
					tmp = np.array(255*(tmp-tmp.min())/(tmp.max()-tmp.min()),dtype=np.uint8)
					imageio.imsave(os.path.join(dir_file,'Outputs','zoom',str(self.shootID),'channel2_'+str(k).zfill(10)+'.png'),tmp)
				self.prev_shoot = self.w1.x
			else:
				print('Not in shooting mode.')
			f.close()

		"""
		Load masks from tiff or png file. User should select a tiff image containing a time sequence of mask, or several png files refered as masks.
		"""
		def loadMasks():
			loadsuccesfull = False
			self.masks_path = QFileDialog.getOpenFileNames(self, "Select file(s) where masks are.",os.getcwd(),
			"*.png *.tiff *.tif")
			if len(self.masks_path[0])!=0:
				if  self.masks_path[0][0].endswith('tiff') or self.masks_path[0][0].endswith('tif'):
					TIFF_masks = True
				elif not(self.masks_path[0][0].endswith('png')):
					TIFF_masks = False
				else:
					TIFF_masks = False
				loadsuccesfull = True
			else:
				TIFF_masks = False
				print("Error - no file found.")
			# load images
			if TIFF_masks:
				masks_path = self.masks_path[0][0]
				print("Warning: only one tiff can be process.")
				mask = np.squeeze(np.array(imageio.mimread(masks_path,memtest=False),dtype=np.uint8))
			else:
				pictList = self.masks_path[0]
				tmp = imageio.imread(pictList[0])
				mask = np.array(bytescale(tmp),dtype=np.uint8)
				mask = np.zeros((len(pictList),mask.shape[0],mask.shape[1]))
				for i in range(len(pictList)):
					tmp = imageio.imread(pictList[i])
					mask[i] = np.array(bytescale(tmp),dtype=np.uint8)
			self.masks_original = mask.copy()
			self.masks = np.array(self.masks_original > 0,dtype=np.uint8)
			if mask.shape[0] != self.finish+1:
				print('Not same number of masks than images.')
				loadsuccesfull = False
			if loadsuccesfull:
				if self.plot_mask != True:
					self.plot_mask = True
					self.p2 = self.win.addPlot(colspan=2)
					self.img_m = pg.ImageItem(None, border="w")
					self.p2.addItem(self.img_m)  
					self.maskLoaded = True
					self.win.show()
					self.w1_m = Slider_thresh(0,1)
					self.horizontalLayout.addWidget(self.w1_m)
					p1.setAspectLocked(ratioKept)
					p1.autoRange()
					self.img_m.setImage(np.rot90(self.masks[self.w1.x],3))
					self.p2.setAspectLocked(ratioKept)
					self.p2.autoRange()
					self.p2.setAspectLocked()
					self.p2.autoRange()
					self.w1_m.valueChangedX.connect(updateThresh)
					updateThresh()
				else:
					self.img_m.setImage(np.rot90(self.masks[self.w1.x],3))
					self.p2.setAspectLocked(ratioKept)
					self.p2.autoRange()
					self.p2.setAspectLocked()
					self.p2.autoRange()
			
		"""
		Compute masks based on neural network model.
		User should select 'h5' or 'hdf5' file. 
		Custom loss is implemented (binary cross entropy + dice coefficient). If any other custom loss 
		should be use, modify in the following function.
		Images are put in [0,1] and then convert to the format needed by the neural network (usually uint16). 
		Masks are saved into 'Outputs' into the folder containing image 
		"""
		def computeMasks():
			from keras.models import load_model
			from tensorflow.python.client import device_lib
			from skimage.transform import resize
			from keras import backend as K
			from keras.losses import binary_crossentropy
			MODE = "GPU" if "GPU" in [k.device_type for k in device_lib.list_local_devices()] else "CPU"
			print(MODE)
			print('########### Computing masks - please wait... ###########')

			def dice_coef_K(y_true, y_pred, smooth=1):
				y_true_f = K.flatten(y_true)
				y_pred_f = K.flatten(y_pred)
				intersection = K.sum(y_true_f * y_pred_f)
				return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
			def dice_coef_loss_K(y_true, y_pred):
				return 1-dice_coef_K(y_true, y_pred)
			self.model = QFileDialog.getOpenFileName(self, "Select h5 file defining the AI model.",os.getcwd(),
			"*.h5 *.hdf5")

			# Load model and parameters
			l1 = 1.
			l2 = -1.
			loss1 = binary_crossentropy
			loss2 = dice_coef_loss_K
			def custom_loss(y_true,y_pred):
				return l1*loss1(y_true,y_pred)+l2*loss2(y_true,y_pred)
			net = load_model(self.model[0],custom_objects={'custom_loss':custom_loss})
			nx = int(net.input.get_shape()[1])
			ny = int(net.input.get_shape()[2])
			TIME = int(net.input.get_shape()[3])
			if LSTM:
				nx = int(net.input.get_shape()[2])
				ny = int(net.input.get_shape()[3])
				TIME = int(net.input.get_shape()[1])
			else:
				nx = int(net.input.get_shape()[1])
				ny = int(net.input.get_shape()[2])
				TIME = int(net.input.get_shape()[3])				
			if TIME>self.im.shape[0]:
				raise ValueError('Require at least {0} images, {1} given. This is important since the segmentation used teporal information.'.format(TIME,self.im.shape[0]))

			# Make data in time batch
			nb_serie = int(self.finish+1)//TIME
			im_batch = np.zeros((nb_serie, nx, ny, TIME), dtype=type_im)
			masks = np.zeros((nx, ny, self.finish+1), dtype=type_im)      
			for i in range(nb_serie):
				tmp = self.im_nn[i*TIME:(i+1)*TIME].copy()
				for t in range(TIME):
					im_batch[i,:,:,t] = resize(tmp[t], (nx,ny), mode='constant', preserve_range=True)
				im_batch[i] = np.array(np.iinfo(type_im).max*(np.array(im_batch[i],dtype=np.float64)-np.min(im_batch[i]))/(np.max(im_batch[i])-np.min(im_batch[i])),dtype=type_im)
			# if nb_serie >0:
			# 	im_batch = np.iinfo(type_im).max*(im_batch- np.min(im_batch))/(np.max(im_batch)-np.min(im_batch))
			im_batch = np.expand_dims(np.array(im_batch, dtype=type_im),4)
			if LSTM:
				im_batch = np.rollaxis(im_batch,3,1)

			# Compute mask for the first time batchs
			masks = np.zeros((self.finish+1,self.im.shape[1],self.im.shape[2]))
			for i in range(nb_serie):
				print("Neural network progress : {0}%.".format(int(100*i/nb_serie)))
				tmp = np.array(np.expand_dims(im_batch[i],0),dtype=np.float32)/np.iinfo(type_im).max
				masks_ = np.array(np.squeeze(net.predict(tmp)),dtype=np.float32)
				# masks_ = np.squeeze(tmp)
				if LSTM:
					# TODO: check this part
					for t in range(TIME):
						masks[i*TIME+t] = resize(masks_[t],(self.im.shape[1],self.im.shape[2]), mode='constant', preserve_range=True)	
				else:
					for t in range(TIME):
						# masks[i*TIME+t] = np.squeeze(self.im[TIME*i+t])
						masks[i*TIME+t] = resize(masks_[:,:,t],(self.im.shape[1],self.im.shape[2]), mode='constant', preserve_range=True)	

			# Compute mask for the remaining images 
			if self.finish != TIME*nb_serie:
				tmp = self.im_nn[self.finish+1-TIME:].copy()
				im_tmp = np.zeros((1,nx,ny,TIME,1))
				for t in range(TIME):
					im_tmp[0,:,:,t,0] = resize(tmp[t], (nx,ny), mode='constant', preserve_range=True)
				im_tmp[0] = np.array(np.iinfo(type_im).max*(np.array(im_tmp[0],dtype=np.float64)-np.min(im_tmp[0]))/(np.max(im_tmp[0])-np.min(im_tmp[0])),dtype=type_im)
				im_tmp = np.array(im_tmp,dtype=np.float32)/np.iinfo(type_im).max
				tmp = np.array(np.squeeze(net.predict(im_tmp)),dtype=np.float32)
				for t in range((self.finish+1-nb_serie*TIME)):
					masks[nb_serie*TIME+t] = resize(tmp[:,:,TIME-(self.finish-nb_serie*TIME)-1+t],(self.im.shape[1],self.im.shape[2]), mode='constant', preserve_range=True)

			# Save masks in same folder than images			
			if not os.path.exists(os.path.join(dir_file,'Outputs','masks')):
				os.makedirs(os.path.join(dir_file,'Outputs','masks'))	
			for i in range(self.finish+1):
				imageio.imsave(os.path.join(dir_file,'Outputs','masks',str(i).zfill(10)+'.png'),masks[i])
			print('Masks computed.')
			self.masks_original = masks.copy()
			self.masks = (self.masks_original > 0).astype(np.uint8)

			if self.plot_mask != True:
				self.plot_mask = True
				self.p2 = self.win.addPlot(colspan=2)
				self.img_m = pg.ImageItem(None, border="w")
				self.p2.addItem(self.img_m)  
				self.maskLoaded = True
				self.win.show()
				self.w1_m = Slider_thresh(0,1)
				self.horizontalLayout.addWidget(self.w1_m)
				p1.setAspectLocked(ratioKept)
				p1.autoRange()
				self.p2.setAspectLocked(ratioKept)
				self.p2.autoRange()
				self.img_m.setImage(np.rot90(self.masks[self.w1.x],3))
				self.p2.setAspectLocked()
				self.p2.autoRange()
				self.w1_m.valueChangedX.connect(updateThresh)
				updateThresh()
			else:
				self.p2.setAspectLocked(ratioKept)
				self.p2.autoRange()
				self.img_m.setImage(np.rot90(self.masks[self.w1.x],3))
				self.p2.setAspectLocked()
				self.p2.autoRange()
				self.w1_m.valueChangedX.connect(updateThresh)

		def updateThresh():
			self.masks = np.array(self.masks_original > self.w1_m.x,dtype=np.uint8)
			updateImage()

		# Get action of user
		self.w2.isShooted.connect(shootingProcedure)
		self.w2.isunShooted.connect(unShoot)
		self.w2.isG1.connect(G1)
		self.w2.isEarlyS.connect(earlyS)
		self.w2.isMidS.connect(midS)
		self.w2.isLateS.connect(lateS)
		self.w2.isG2.connect(G2)
		self.w2.isEnd.connect(end)
		self.w2.isloadMasks.connect(loadMasks)
		self.w2.iscomputeMasks.connect(computeMasks)
		self.img = pg.ImageItem(None, border="w")
		# self.img.setRect(QRect(100, 200, 11, 16))
		p1.addItem(self.img)

		# Custom ROI for selecting an image region (axis swaped)
		self.roi = pg.ROI([20, 20], [int(self.ny/10), int(self.nx/10)])
		self.roi.addScaleHandle([0.5, 1], [0.5, 0.001])
		self.roi.addScaleHandle([0, 0.5], [0.999, 0.5])
		self.roi.addScaleHandle([0.5, 0], [0.5, 0.999])
		self.roi.addScaleHandle([1, 0.5], [0.001, 0.5])

		# update when user change view
		def updateImage():
			self.data = self.im[self.w1.x]
			self.img.setImage(np.rot90(self.data,3))
			if self.maskLoaded:
				self.img_m.setImage(np.rot90(self.masks[self.w1.x],3))
		self.w1.valueChangedX.connect(updateImage)
		p1.addItem(self.roi)
		self.roi.setZValue(10)  # make sure ROI is drawn above image

		# Contrast/color control
		hist = pg.HistogramLUTItem()
		hist.setImageItem(self.img)
		self.win.addItem(hist)
		self.img.setImage(np.rot90(self.data,3))
		hist.setLevels(self.data.min(), self.data.max())	
		p1.setAspectLocked()
		p1.autoRange()  

		self.boundingBox = [None,None]
		# Callbacks for handling user interaction
		def updatePlot():
			global img, roi, data
			selected = self.roi.getArrayRegion(self.data, self.img)
			self.boundingBox = [[int(self.roi.pos().x()),int(self.roi.pos().y())],\
			[int(self.roi.pos().x()+self.roi.size().x()),int(self.roi.pos().y()+self.roi.size().y())]]
		self.roi.sigRegionChanged.connect(updatePlot)
		updatePlot()


		self.w3 = InterfaceManagerButton()
		self.horizontalLayout.addWidget(self.w3)
		self.w3.isQuit.connect(quitProcedure)
		self.w3.isChannel.connect(newChannelProcedure)

# Interpret image data as row-major instead of col-major
# pg.setConfigOptions(imageAxisOrder='row-major')

# pg.mkQApp()
# win = pg.GraphicsLayoutWidget()
# win.setWindowTitle('pyqtgraph example: Image Analysis')
if __name__ == '__main__':
	app = QApplication(sys.argv)
	w = Widget()
	w.show()
	sys.exit(app.exec_())