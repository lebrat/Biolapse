# ![ ](https://github.com/lebrat/Biolapse/blob/master/header.png "Biolapse")

[![Build Status](https://travis-ci.org/lebrat/Biolapse.svg?branch=master)](https://travis-ci.org/lebrat/Biolapse)

Biolapse is an open source graphical user interface dedicated to segment and track biological objects. All the details about the implementation is detail in ...



## Getting Started

This project is written in python 3 and make use of [TensorFlow](https://www.tensorflow.org/) for the segmenting task and [PyQtGraph](http://pyqtgraph.org/) and [PyQt5](https://pypi.org/project/PyQt5/) for the graphical interface.

### Installation
To install this project and all its prerequisites simply type in your terminal :

`pip install git+https://github.com/lebrat/Biolapse`

now you can clone the project :

`git clone https://github.com/lebrat/Biolapse.git`

to launch the main interface open a terminal in the Biolapse's directory and type :

`python GUI.py`

### Tutorial video 
[![Tutorial](https://img.youtube.com/vi/nomVideo/0.jpg)](https://www.youtube.com/watch?v=nomVideo)

### Traning the neural networks

The providing neural network can be retrained on **your** dataset using the command :

`python train.py`

- [ ] Describe how to add dataset.


## Authors
This software was developped by [Valentin Debarnot](https://scholar.google.fr/citations?user=gxBQ7d4AAAAJ&hl=fr) and [Léo Lebrat](lebrat.org) members of the [Toulouse institute of Mathematics](https://www.math.univ-toulouse.fr/?lang=en) France. It is based on an original idea by [Thomas Mangeat](https://scholar.google.com/citations?user=hPebN5AAAAAJ&hl=fr)

## License

This software is open source distributed under license MIT.