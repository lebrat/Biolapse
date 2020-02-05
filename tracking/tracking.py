import cv2
import numpy as np

"""
Given a mask (binary image), draw the border of the mask on the image in red.
"""
def draw_border(mask, image):
	contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return cv2.drawContours(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB), contours, -1, (255, 0, 0), 5)


"""
Find location of the minimum of CF to find the mask that is more likely to be associated to barycenter.
The cost function CF is defined by:
.......

"""
def argmin_CF(localCells,barycenter,vol,gamma):
	minCost = np.inf
	indexMin = -1		
	if len(localCells)==0:
		print('No mask found.')
	for ind,elmt in enumerate(localCells):
		l1 = np.linalg.norm(elmt[1] - barycenter) # distance between barycenter
		l2 = np.linalg.norm(float(elmt[2])-float(vol))/float(vol)
		cost = gamma[0]*l1 + gamma[1]*l2
		if cost < minCost:
			indexMin = ind
			minCost = cost
	return indexMin, minCost



def argmin_first(localCells,barycenter):
	minCost = np.inf
	indexMin = -1		
	if len(localCells)==0:
		print('No mask found.')
	for ind,elmt in enumerate(localCells):
		l1 = np.linalg.norm(elmt[1] - barycenter) # distance between barycenter
		cost = l1
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
	num,compo=cv2.connectedComponents(binary,connectivity=8)
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
def extractMaskFromPoint(masks, im, im_channel, imageStart, pos, finish, progressbar, minimalSize=25, gamma=[0.5,0.5]):
	if len(im_channel.shape) == 1:
		secondChannel = False
	else:
		secondChannel = True
	#Initialize the tracking and get paramters 
	crtImage = imageStart
	cpt_im = 0
	maskFinal = np.zeros_like(masks[crtImage])
	maskFinal_= np.zeros((finish+1,masks.shape[1],masks.shape[2]))
	barycenter = np.copy(pos)
	nx_mask = masks.shape[1]
	ny_mask = masks.shape[2]
	barycenter[0] = nx_mask - barycenter[0] # need to adapt
	DictBar = {} # contains barycenter and image index 
	DictBar[crtImage] = barycenter.copy()
	im_crop = {} # crop of masks
	mask_crop = {} # crop of masks
	im_crop2 = {} # crop of masks is second channel of interest
	im_out = np.repeat(np.expand_dims(im.copy(),3),3,3) # Need to turn into rgb image, HistogramLUTItem doesn't support rgb
	# im_out = np.zeros_like(im)
	min_x = np.zeros(finish-imageStart+1)
	max_x = np.zeros(finish-imageStart+1)
	min_y = np.zeros(finish-imageStart+1)
	max_y = np.zeros(finish-imageStart+1)

	localCells,compo = find_connected_components(masks[crtImage],minimalSize) # Detect connected components
	indexMin = argmin_first(localCells,barycenter) # Compute argmin CF
	maskFinal[np.where(compo==localCells[indexMin][0])] = 1 # take only mask of interest

	# mask_ = np.array(255*(maskFinal>1e-7),dtype=np.uint8)
	# # Erode
	# kernel = np.ones((5,5), np.uint8) 
	# mask_erosion = cv2.erode(mask_, kernel, iterations=1) 
	# # Dilate
	# kernel = np.ones((4,4), np.uint8) 
	# maskFinal = cv2.dilate(mask_erosion, kernel, iterations=1) 
	# volume = np.sum(maskFinal)
	
	# Select region of interest to save crops
	delta = 2
	r = np.where(maskFinal)
	max_x[cpt_im] = np.min((np.max(r[0])+delta,im.shape[1]))
	max_y[cpt_im] = np.min((np.max(r[1])+delta,im.shape[2]))
	min_x[cpt_im] = np.max((np.min(r[0])-delta,0))
	min_y[cpt_im] = np.max((np.min(r[1])-delta,0))

	im_out[crtImage] = draw_border(maskFinal,im[crtImage].astype(np.uint8)) # HistogramLUTItem doesn't support rgb
	# im_out[crtImage] = np.dot(draw_border(maskFinal,im[crtImage].astype(np.uint8)), [0.299, 0.587, 0.144])
	maskFinal_[crtImage] = maskFinal
	# Repeat the operation for each time step
	# progressbar.setValue(crtImage)
	
	## compute outputs
	cpt_im = 0
	max_x_axis = np.max(max_x-min_x)
	max_y_axis = np.max(max_y-min_y)
	add_x = max_x_axis - (max_x[cpt_im]-min_x[cpt_im])
	add_y = max_y_axis - (max_y[cpt_im]-min_y[cpt_im])
	x_l = int(min_x[cpt_im]-np.floor(add_x/2))
	x_u = int(max_x[cpt_im]+np.ceil(add_x/2))
	y_l = int(min_y[cpt_im]-np.floor(add_y/2))
	y_u = int(max_y[cpt_im]+np.ceil(add_y/2))
	if x_l < 0:
		x_u -= x_l
		x_l = 0
	if y_l < 0:
		y_u -= y_l
		y_l = 0
	if x_u >nx_mask-1:
		x_l -= x_u - (nx_mask -1)
		x_u = nx_mask-1
	if y_u >ny_mask-1:
		y_l -= y_u - (ny_mask -1)
		y_u = ny_mask-1
	im_crop[crtImage] = im[crtImage,x_l:x_u,y_l:y_u]
	mask_crop[crtImage] = maskFinal[x_l:x_u,y_l:y_u]
	if secondChannel:
		im_crop2[crtImage] = im_channel[crtImage,x_l:x_u,y_l:y_u]
	cpt_im += 1

	# cost = np.zeros(finish-imageStart)
	while crtImage < finish :
		crtImage += 1
		barycenter = np.array(localCells[indexMin][1]) # restart from barycenter of previous mask.
		DictBar[crtImage] = barycenter.copy()

		maskFinal = np.zeros_like(masks[crtImage])
		localCells,compo = find_connected_components(masks[crtImage],minimalSize) # Detect connected components
		indexMin, c = argmin_CF(localCells,barycenter,volume,gamma) # Compute argmin CF
		maskFinal[np.where(compo==localCells[indexMin][0])] = 1 # take only mask of interest

		# mask_ = np.array(255*(maskFinal>1e-7),dtype=np.uint8)
		# # Erode
		# kernel = np.ones((5,5), np.uint8) 
		# mask_erosion = cv2.erode(mask_, kernel, iterations=1) 
		# # Dilate
		# kernel = np.ones((4,4), np.uint8) 
		# maskFinal = cv2.dilate(mask_erosion, kernel, iterations=1) 

		volume = np.sum(maskFinal)
		# cost[crtImage-1]=c

		# selct image focus
		delta = 2
		r = np.where(maskFinal)
		max_x[cpt_im] = np.min((np.max(r[0])+delta,im.shape[1]))
		max_y[cpt_im] = np.min((np.max(r[1])+delta,im.shape[2]))
		min_x[cpt_im] = np.max((np.min(r[0])-delta,0))
		min_y[cpt_im] = np.max((np.min(r[1])-delta,0))

		im_out[crtImage] = draw_border(maskFinal,im[crtImage].astype(np.uint8)) # HistogramLUTItem doesn't support rgb
		# im_out[crtImage] = np.dot(draw_border(maskFinal,im[crtImage].astype(np.uint8)), [0.299, 0.587, 0.144])
		maskFinal_[crtImage] = maskFinal

		add_x = max_x_axis - (max_x[cpt_im]-min_x[cpt_im])
		add_y = max_y_axis - (max_y[cpt_im]-min_y[cpt_im])
		x_l = int(min_x[cpt_im]-np.floor(add_x/2))
		x_u = int(max_x[cpt_im]+np.ceil(add_x/2))
		y_l = int(min_y[cpt_im]-np.floor(add_y/2))
		y_u = int(max_y[cpt_im]+np.ceil(add_y/2))
		if x_l < 0:
			x_u -= x_l
			x_l = 0
		if y_l < 0:
			y_u -= y_l
			y_l = 0
		if x_u >nx_mask-1:
			x_l -= x_u - (nx_mask -1)
			x_u = nx_mask-1
		if y_u >ny_mask-1:
			y_l -= y_u - (ny_mask -1)
			y_u = ny_mask-1
		# import ipdb; ipdb.set_trace()
		im_crop[crtImage] = im[crtImage,x_l:x_u,y_l:y_u]
		mask_crop[crtImage] = maskFinal[x_l:x_u,y_l:y_u]
		if secondChannel:
			im_crop2[crtImage] = im_channel[crtImage,x_l:x_u,y_l:y_u]
		cpt_im += 1

	return im_out.astype(np.uint8), DictBar, mask_crop, im_crop, im_crop2, maskFinal_
