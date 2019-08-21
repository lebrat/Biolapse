# ![ ](https://github.com/lebrat/Biolapse/blob/master/header.png "Biolapse")

[![Build Status](https://travis-ci.org/lebrat/Biolapse.svg?branch=master)](https://travis-ci.org/lebrat/Biolapse)  [![DOI](https://zenodo.org/badge/202534809.svg)](https://zenodo.org/badge/latestdoi/202534809)


Biolapse is an open source graphical user interface dedicated to segment and track biological objects. All the details about the implementation are detailed in ... All experiments from this document can be reproduced using the two commands :

`python script_train_XP_paper.py` and `python script_post_process_XP_paper.py`.

## Getting Started

This project is written in python 3 and makes use of [TensorFlow](https://www.tensorflow.org/) for the segmenting task and [PyQtGraph](http://pyqtgraph.org/) and [PyQt5](https://pypi.org/project/PyQt5/) for the graphical interface.

### Installation
To install all the prerequisites simply type in your terminal :

`pip install git+https://github.com/lebrat/Biolapse`

now you can clone the project :

`git clone https://github.com/lebrat/Biolapse.git`

to launch the main interface open a terminal in the Biolapse's directory and type :

`python GUI.py`

### Tutorial video 
[![Tutorial](https://img.youtube.com/vi/nomVideo/0.jpg)](https://www.youtube.com/watch?v=nomVideo)

### Traning the neural network

The provided neural network can be retrained on **your** dataset using the command :

`python script_train.py`

- Trainning data must be placed at segmentation/Data/png, with images at segmentation/Data/png/train_im and masks at segmentation/Data/png/train/mask.
- Testing data must be placed at segmentation/Data/png, with images at segmentation/Data/png/test_im and masks at segmentation/Data/png/test/mask. 
Images and masks must be in png format.

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
