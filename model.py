#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 13:03:21 2018

@author: kiyoshitaro
"""

import keras
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.regularizers import *


def get_model():
	aliases = {}
	Input_4 = Input(shape=(499,), name='Input_4')
	Embedding_1 = Embedding(name='Embedding_1',output_dim= 128,dropout= 0.2,input_dim= 499)(Input_4)
	Convolution1D_1 = Convolution1D(name='Convolution1D_1',nb_filter= 50,filter_length= 5,activation= 'relu' )(Embedding_1)
	MaxPooling1D_1 = MaxPooling1D(name='MaxPooling1D_1')(Convolution1D_1)
	Flatten_1 = Flatten(name='Flatten_1')(MaxPooling1D_1)
	Dropout_12698 = Dropout(name='Dropout_12698',p= 0.5)(Flatten_1)
	Dense_31266 = Dense(name='Dense_31266',output_dim= 10,activation= 'relu' )(Dropout_12698)
	Dense_31267 = Dense(name='Dense_31267',output_dim= 1,activation= 'sigmoid' )(Dense_31266)

	model = Model([Input_4],[Dense_31267])
	return aliases, model


from keras.optimizers import *

def get_optimizer():
	return Adam()

def is_custom_loss_function():
	return False

def get_loss_function():
	return 'binary_crossentropy'

def get_batch_size():
	return 32

def get_num_epoch():
	return 15

def get_data_config():
	return '{"samples": {"test": 0, "training": 20000, "validation": 5000, "split": 1}, "shuffle": true, "datasetLoadOption": "batch", "kfold": 1, "dataset": {"samples": 25000, "type": "public", "name": "imdb"}, "numPorts": 1, "mapping": {"Sentiment": {"port": "OutputPort0", "options": {"Scaling": 1, "Normalization": false}, "shape": "", "type": "Numeric"}, "Review": {"port": "InputPort0", "options": {"Scaling": 1, "Normalization": false}, "shape": "", "type": "Array"}}}'