#!/usr/bin/python

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.metrics import mean_squared_error
import os
import csv

#setting
datafile = 'studentperform.csv'
studentmodel = 'studentmodel.h5'
batch_size = 10
hidden_neuron = 10
trainsize = 900
iterasi = 200

def generatemodel(totvar):
	# create and fit the LSTM network
	model = Sequential()
	model.add(Dense(3, batch_input_shape=(batch_size, totvar), activation='sigmoid'))
	model.add(Dense(hidden_neuron, activation='sigmoid'))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

#read data
alldata = np.genfromtxt(datafile,delimiter=',')[1:]

#separate between training and test
trainparam = alldata[:900, :-1]
trainlabel = alldata[:900, -1]

testparam = alldata[900:, :-1]
testlabel = alldata[900:, -1]

trainparam = trainparam[len(trainparam)%10:]
trainlabel = trainlabel[len(trainlabel)%10:]

testparam = testparam[len(testparam)%10:]
testlabel = testlabel[len(testlabel)%10:]


###############
#normalization#
###############

trainparamnorm = np.zeros(np.shape(trainparam))
trainlabelnorm = np.zeros(np.shape(trainlabel))

testparamnorm = np.zeros(np.shape(testparam))
testlabelnorm = np.zeros(np.shape(testlabel))

print 'shape label adalah', np.shape(testlabelnorm)

#for param
for i in xrange(len(trainparam[0])-2):
	trainparamnorm[:,i] = (trainparam[:,i] - np.min(trainparam[:,i])) / (np.max(trainparam[:,i]) - np.min(trainparam[:,i]))
	testparamnorm[:,i] = (testparam[:,i] - np.min(trainparam[:,i])) / (np.max(trainparam[:,i]) - np.min(trainparam[:,i]))

for i in xrange(2):
	trainparamnorm[:,-2+i] = (trainparam[:,-2+i] - 0.0) / (20.0 - 0.0)
	testparamnorm[:,-2+i] = (testparam[:,-2+i] - 0.0) / (20.0 - 0.0)

#for label
trainlabelnorm = (trainlabel - np.min(trainlabel)) / (np.max(trainlabel) - np.min(trainlabel))
testlabelnorm = (testlabel - np.min(trainlabel)) / (np.max(trainlabel) - np.min(trainlabel))

###############
#running model#
###############

mod = generatemodel(len(trainparamnorm[0]))
mod.fit(trainparamnorm, trainlabelnorm, epochs=iterasi, batch_size=batch_size, verbose=2, shuffle=True)

#save trained model
mod.save(studentmodel)
