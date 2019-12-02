# ![ ](https://github.com/lebrat/Biolapse/blob/master/header.png "Biolapse")

[![Build Status](https://travis-ci.org/lebrat/Biolapse.svg?branch=master)](https://travis-ci.org/lebrat/Biolapse)  [![DOI](https://zenodo.org/badge/202534809.svg)](https://zenodo.org/badge/latestdoi/202534809)


Biolapse is an open source graphical user interface dedicated to segment and track biological objects. All the details about the implementation are detailed in ... All experiments from this document can be reproduced using the two commands :

`python script_train_XP_paper.py` and `python script_post_process_XP_paper.py`.

## Getting Started

This project is written in python 3 and makes use of [TensorFlow](https://www.tensorflow.org/) for the segmenting task and [PyQtGraph](http://pyqtgraph.org/) and [PyQt5](https://pypi.org/project/PyQt5/) for the graphical interface.

![LM Pipeline](workflow.png)

### Installation
To install all the prerequisites simply type in your terminal :

`pip install git+https://github.com/lebrat/Biolapse`

now you can clone the project :

`git clone https://github.com/lebrat/Biolapse.git`

to launch the main interface open a terminal in the Biolapse's directory and type :

`python GUI.py`

### Tutorial video 
[![Tutorial](https://img.youtube.com/vi/nomVideo/0.jpg)](https://www.youtube.com/watch?v=nomVideo)

## Segmentation: traning the neural network

The provided neural network can be retrained on **your** dataset.
Place the tif images in 

`Data/Acquisitions/Train`

and 

`Data/Acquisitions/Train`

Execute the script **script_preprocess_data** for save the images in png format and to artificially augment the data. Simply use the following command in a terminal :

`python script_preprocess_data.py`

The training of the neural network is then straight forward: 

`python script_train.py`

The neural network used can be changed by affecting the variable **model_name** with **Unet3D,**, **Unet2D** or **LSTM**.

You can finally visualize the training of your neural network with the command : 

`python script_post_process.py`


## Classification: labeling training data-set

Use the GUI interfac to label cells.

Launch the GUI in **tracking** directory with 

`python GUI.py`


## Classification: training a neural network

In order to run the classification algorithm which determines the phase of the S phase from cropped cells, we need ton train a neural network to return probability of being in each state.

Execute the script in **classification** directory with

`python script_train_classification.py`

to train the neural network autoencoder to predict probability of each state.


### Evaluating the neural network

If you dont want or can't re-train the neural networks, you can download them at :
[![Download pre-trained neural networks](https://drive.google.com/drive/folders/19YDvcw3C33yNX0c8HSmI__JRfB21YIW9?usp=sharing)


The trained neural network can be evaluated on the test data set using the command :

`python script_post_process.py`

You can also directly apply the pretrained neural network by using the same command.

## Authors
This software was developped by [Valentin Debarnot](https://sites.google.com/view/debarnot/) and [Léo Lebrat](lebrat.org) members of the [Toulouse institute of Mathematics](https://www.math.univ-toulouse.fr/?lang=en) France. It is based on an original idea by [Thomas Mangeat](https://scholar.google.com/citations?user=hPebN5AAAAAJ&hl=fr).

## License

This software is open source distributed under license MIT.
