import cv2
import numpy as np

"""
Given a mask (binary image), draw the border of the mask on the image in red.
"""
def draw_border(mask, image):
	contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return cv2.drawContours(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB), contours, -1, (255, 0, 0), 1)


"""
Find location of the minimum of CF to find the mask that is more likely to be associated to barycenter.
The cost function CF is defined by:
.......

"""
def argmin_CF(localCells,barycenter):
	minCost = np.inf
	indexMin = -1		
	if len(localCells)==0:
		print('No mask found.')
	for ind,elmt in enumerate(localCells):
		cost = np.linalg.norm(elmt[1] - barycenter)
		if cost < minCost:
			indexMin = ind
			minCost = cost
	return indexMin

"""
Detect connected components with minimal area defined by 'minimalSize'.
Return list containing the number of the component, its barycenter and its area. It also returns 
'compo', an image where each connected componenent is defined by the number given in localCells.
"""
def find_connected_components(masks,minimalSize):
	binary = cv2.threshold(masks, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	num,compo=cv2.connectedComponents(binary)
	localCells = []
	for chunckNo in range(1,num):
		if len(np.where(compo==chunckNo)[0]) < minimalSize:
			compo[np.where(compo==chunckNo)] = 0
		else:
			localCells += [[chunckNo,(np.where(compo==chunckNo)[0].mean(),np.where(compo==chunckNo)[1].mean()),len(np.where(compo==chunckNo)[0])]]
	return localCells, compo


"""
Track cell associated to barycenter given by 'pos', return original sequence with mask border draw in red.

Inputs:
 - masks: sequence of masks associated to the original image.
 - im: sequence of original images.
 - im_channel: sequence of other channel of ithe images. If none, set to [].
 - pos: position of the first barycenter.
 - finish: time until the tracking is needed.
 - progressbar: progressbar to visualize progress.
 - minimal_size: minimal area of the masks.

Outputs:
 - im_out: original sequence of images with mask border draw in red.
 - DictBar: contains sequence of barycenters.
 - im_crop: sequence of crop from the image centered around the region of interest.
 - m_crop2: sequence of crop from another channel from original image, if exists, centered around the region of interest.
"""
def extractMaskFromPoint(masks, im, im_channel, imageStart, pos, finish, progressbar, minimalSize=25):
	if len(im_channel.shape) == 1:
		secondChannel = False
	else:
		secondChannel = True
	#Initialize the tracking and get paramters 
	crtImage = imageStart
	maskFinal = np.zeros_like(masks[crtImage])
	barycenter = np.copy(pos)
	nx_mask = masks.shape[1]
	ny_mask = masks.shape[2]
	barycenter[0] = nx_mask - barycenter[0] # need to adapt
	DictBar = {} # contains barycenter and image index 
	DictBar[crtImage] = barycenter.copy()
	im_crop = {} # crop of masks
	im_crop2 = {} # crop of masks is second channel of interest
	im_out = np.repeat(np.expand_dims(im.copy(),3),3,3) # Need to turn into rgb image

	localCells,compo = find_connected_components(masks[crtImage],minimalSize) # Detect connected components
	indexMin = argmin_CF(localCells,barycenter) # Compute argmin CF
	maskFinal[np.where(compo==localCells[indexMin][0])] = 1 # take only mask of interest
	
	# Select region of interest to save crops
	delta = 2
	r = np.where(maskFinal)
	max_x = np.min((np.max(r[0])+delta,im.shape[1]))
	max_y = np.min((np.max(r[1])+delta,im.shape[2]))
	min_x = np.max((np.min(r[0])-delta,0))
	min_y = np.max((np.min(r[1])-delta,0))
	im_crop[crtImage] = im[crtImage,min_x:max_x,min_y:max_y]
	if secondChannel:
		im_crop2[crtImage] = im_channel[crtImage,min_x:max_x,min_y:max_y]
	
	im_out[crtImage] = draw_border(maskFinal,im[crtImage].astype(np.uint8))
	# Repeat the operation for each time step
	progressbar.setValue(crtImage)
	while crtImage < finish :
		crtImage += 1
		barycenter = np.array(localCells[indexMin][1]) # restart from barycenter of previous mask.
		DictBar[crtImage] = barycenter.copy()

		maskFinal = np.zeros_like(masks[crtImage])
		localCells,compo = find_connected_components(masks[crtImage],minimalSize) # Detect connected components
		indexMin = argmin_CF(localCells,barycenter) # Compute argmin CF
		maskFinal[np.where(compo==localCells[indexMin][0])] = 1 # take only mask of interest

		# selct image focus
		delta = 2
		r = np.where(maskFinal)
		max_x = np.min((np.max(r[0])+delta,im.shape[1]))
		max_y = np.min((np.max(r[1])+delta,im.shape[2]))
		min_x = np.max((np.min(r[0])-delta,0))
		min_y = np.max((np.min(r[1])-delta,0))
		im_crop[crtImage] = im[crtImage,min_x:max_x,min_y:max_y]
		if secondChannel:
			im_crop2[crtImage] = im_channel[crtImage,min_x:max_x,min_y:max_y]

		im_out[crtImage] = draw_border(maskFinal,im[crtImage].astype(np.uint8))

		progressbar.setValue(crtImage)
	return im_out.astype(np.uint8), DictBar, im_crop, im_crop2