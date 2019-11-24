import numpy as np
import re
from sys import argv
import sys
import csv
import glob
import os
import cv2
import operator


"""
Convert a rgb image to grayscale.
"""
def rgb_to_gray(rgb):
    return np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 

"""
Order strings in natural order.
"""
def stringSplitByNumbers(x):
    r = re.compile('(\d+)')
    l = r.split(x)
    return [int(y) if y.isdigit() else y for y in l]

"""
Remove duplicated line from csv file saved by Biolapse interface.
"""
def remove_duplicate_csv(source):
	# # Create the output file as input with _deduped before .csv extension
	# source = argv[1]
	destination = source.replace('.csv', '_deduped.csv')
	data = open(source, 'r')
	target = open(destination, 'w')
	# Let the user know you are starting, in case you are de-dupping a huge file 
	print("\nRemoving duplicates from %r" % source)

	# Initialize variables and counters
	unique_lines = set()
	source_lines = 0
	duplicate_lines = 0

	# Loop through data, write uniques to output file, skip duplicates.
	for line in data:
		source_lines += 1
		# Strip out the junk for an easy set check, also saves memory
		line_to_check = line.strip('\r\n')	
		if line_to_check in unique_lines: # Skip if line is already in set
			duplicate_lines += 1
			continue 
		else: # Write if new and append stripped line to list of seen lines
			target.write(line)
			unique_lines.add(line_to_check)

	# Be nice and close out the files
	target.close()
	data.close()

    
"""
Extract information from csv file saved by Biolapse interface.
"""
def get_labels(path_to_crop):
    # Load labels
    all_img = glob.glob(os.path.join(path_to_crop, 'zoom','*', '*.png'))
    all_img = sorted(all_img,key=stringSplitByNumbers)
    labels = []
    idx = []
    csv_file_ = os.path.join(path_to_crop, 'cells.csv')
    remove_duplicate_csv(csv_file_)
    csv_file_ = os.path.join(path_to_crop, 'cells_deduped.csv')
    with open(csv_file_) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        name_cell_ = ''
        first_run = True
        for row in csv_reader:
            idx_matching = [k for k in range(
                len(all_img)) if name_cell_ in all_img[k]]
            name_cell = row[0]
            if name_cell == name_cell_:
                # add previous cell labels
                if not(row[1] == ' '):
                    crt_time = int(row[1])
                    for k in range(prev_time, crt_time):
                        lab_cell.append(0)
                    prev_time = crt_time
                elif not(row[3] == ' '):
                    crt_time = int(row[3])
                    for k in range(prev_time, crt_time):
                        lab_cell.append(0)
                    prev_time = crt_time
                elif not(row[5] == ' '):
                    crt_time = int(row[5])
                    for k in range(prev_time, crt_time):
                        lab_cell.append(1)
                    prev_time = crt_time
                elif not(row[7] == ' '):
                    # import ipdb; ipdb.set_trace()
                    crt_time = int(row[7])
                    for k in range(prev_time, crt_time):
                        lab_cell.append(2)
                    prev_time = crt_time
                elif not(row[9] == ' '):
                    crt_time = int(row[9])
                    for k in range(prev_time, crt_time):
                        lab_cell.append(3)
                    prev_time = crt_time
                elif not(row[11] == ' '):
                    crt_time = int(row[11])
                    for k in range(prev_time, crt_time+1):
                        lab_cell.append(4)
                    prev_time = crt_time
                    # lab_cell.append(-1)
                    if not(len(lab_cell) == int(row[11])-start_time+1):
                        print('Warning: inconsistance in time acquisition.')
                    if len(idx_matching) != len(lab_cell):
                        # import ipdb; ipdb.set_trace()
                        print(
                            'Error: not same number of cell than number of labels found.')
                else:
                    print('Warning: no phase mentionned.')
                    print(row)
                    name_cell_ = ''
            else:
                # add corresponding index in image sequence
                if not(first_run):
                    if len(lab_cell) != 0:
                        labels.append(lab_cell)
                        idx.append(idx_matching)
                # create new label list
                lab_cell = []
                if not(row[1] == ' '):
                    start_time = int(row[1])
                    prev_time = start_time
                    first_run = False
                elif not(row[3] == ' '):
                    start_time = int(row[3])
                    prev_time = start_time
                    first_run = False
                elif not(row[5] == ' '):
                    start_time = int(row[5])
                    prev_time = start_time
                    first_run = False
                elif not(row[7] == ' '):
                    start_time = int(row[7])
                    prev_time = start_time
                    first_run = False
                elif not(row[9] == ' '):
                    start_time = int(row[9])
                    prev_time = start_time
                    first_run = False
                elif not(row[11] == ' '):
                    print('Warning: end before it starts.')
                    name_cell_ = ''
                    # start_time = int(row[1])
                    # lab_cell.append(-1)
                else:
                    print('Warning: no phase mentionned.')
                    print(row)
                    name_cell_ = ''
            name_cell_ = name_cell
        idx_matching = [k for k in range(
            len(all_img)) if name_cell_ in all_img[k]]
        if len(lab_cell) != 0:
            labels.append(lab_cell)
            idx.append(idx_matching)
    return labels, idx

## NN
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms
import imageio

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(42050, 500)
        self.fc2 = nn.Linear(500, 4)

    def forward(self, x):
        x = torch.squeeze(x,0)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train(model, device, train_loader, test_loader, optimizer, epoch, nit=100, nbatch=20):
    model.train()
    
    data, target = train_loader
    data_test, target_test = test_loader
    nbImages = data.shape[0]
    for batch_idx in range(nit):
        randi = np.random.randint(0,nbImages,size=nbatch)
        dataloc, targetloc = data[randi,:,:,:] ,target[randi]
        dataloc.to(device)
        targetloc.to(device)

        optimizer.zero_grad()
        output = model(dataloc)
        loss = F.binary_cross_entropy(output,targetloc)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, nit,
                100. * batch_idx / nit, loss.item()))
    # # Test
    randi = np.random.randint(0,data_test.shape[0],size=nbatch)
    dataloc_test, targetloc_test = data_test[randi,:], target_test[randi,:]
    dataloc_test.to(device)
    targetloc_test.to(device)
    output = model(dataloc_test)
    loss = F.binary_cross_entropy(output,targetloc_test)
    print('Test loss: {:.6f}'.format(loss.item()))



class Dataset():

    """Dataset class
        Attributes:
        dataPath (TYPE): Position of the folder containing all the images
        lenDataset (TYPE): Number of images present in this folder
    """
    
    def __init__(self, dataPath,lenDataset):
        'Initialization'
        self.dataPath = dataPath
        self.lenDataset = lenDataset

    def __len__(self):
        'Denotes the total number of samples'
        return self.lenDataset

    def __getitem__(self, index):
        'Generates one sample of data'
        images = np.zeros((self.lenDataset,1,128,128))
        labs = np.load(self.dataPath+os.sep+'feature.npy')
        labs = labs[:self.lenDataset]
        for i in range(self.lenDataset):
            images[i,0,:,:] = np.array(imageio.imread(self.dataPath+os.sep+str(i).zfill(5)+'.png'),dtype=float)/255.

        return torch.Tensor(images), torch.Tensor(labs)
    def getALLDATA(self):
        images = np.zeros((self.lenDataset,1,128,128))
        labs = np.load(self.dataPath+os.sep+'feature.npy')
        labs = labs[:self.lenDataset]
        for i in range(self.lenDataset):
            images[i,0,:,:] = np.array(imageio.imread(self.dataPath+os.sep+str(i).zfill(5)+'.png'),dtype=float)/255.

        return (torch.Tensor(images).cuda(), torch.Tensor(labs).cuda())





"""
Viterbi algorithm to predict the most likely Markov chain based on the probabilities of each time.

INPUT:
    - proba_pred: array of size (nSamples,nStates). nSamples is the number of sample and nStates is the number of possible states. Contains
    probability of each sample to be in each states.
    - nclasses: number of phase to predict.
    - transition: average probability of transition between classes. This possibly encode the restriction on the state path.
"""
def viterbi(proba_pred, nclasses, transition):
    # Check shape
    nSamples = proba_pred.shape[0]
    nStates = proba_pred.shape[1]
    bestPath = np.zeros(nSamples)
    psi = np.zeros((nStates,nSamples))
    c = np.zeros(nSamples)
    viterbi = np.zeros((nStates, nSamples))

    viterbi[:, 0] = proba_pred[0].T
    c[0] = 1.0/np.sum(viterbi[:, 0])
    viterbi[:, 0] = c[0] * viterbi[:, 0]  # apply the scaling factor
    psi[0] = 0

    for t in range(1, nSamples):
        for s in range(0, nStates):
            trans_p = viterbi[:, t-1] * transition[:, s]
            psi[s, t], viterbi[s, t] = max(
                enumerate(trans_p), key=operator.itemgetter(1))
            # viterbi[s,t] = viterbi[s,t]*probaPredictor(model,featMark[t])[s]
            viterbi[s,t] = viterbi[s,t]*proba_pred[t,s]

        if np.sum(viterbi[:,t])==0:
            import ipdb; ipdb.set_trace()
        c[t] = 1.0/np.sum(viterbi[:,t]) 
        viterbi[:,t] = c[t] * viterbi[:,t]

    bestPath[nSamples-1] =  viterbi[:,nSamples-1].argmax() 
    for t in range(nSamples-1,0,-1): 
        bestPath[t-1] = psi[int(bestPath[t]),t]

    return bestPath


def buildTransitionMatrix(vectorEpsilon):
    n = vectorEpsilon.shape[0]
    transitionMat = np.zeros([n,n])
    for i in range(n-1):
        transitionMat[i,i] = 1 - vectorEpsilon[i]
        transitionMat[i,i+1] = vectorEpsilon[i]

    # transitionMat[n-1,n-1] = 1
    transitionMat[n-1,n-1] = 1-vectorEpsilon[-1]
    transitionMat[n-1,0] = vectorEpsilon[-1]
    return transitionMat



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