#!/usr/bin/python

import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, LSTM, RepeatVector, TimeDistributed
from keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import random
import os
import csv

#setting
datafile = 'studentperform.csv'
studentmodel = 'studentmodel.h5'
batch_size = 10
hidden_neuron = 10
trainsize = 900
iterasi = 200
randsample = 100

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

trainparamnorm = np.zeros(np.shape(trainparam)).astype('float32')
trainlabelnorm = np.zeros(np.shape(trainlabel)).astype('float32')

testparamnorm = np.zeros(np.shape(testparam)).astype('float32')
testlabelnorm = np.zeros(np.shape(testlabel)).astype('float32')

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


#load trained model
mod = load_model(studentmodel)

G3pred = mod.predict(testparamnorm, batch_size=batch_size)
G3real = G3pred*20.0

errreal = mean_squared_error(testlabel, G3real)
print 'our error value is', errreal

################################
#permutation importance session#
################################

permutsample = np.zeros((randsample, len(testparamnorm[0])))
for trying in xrange(randsample):
	randval = np.zeros((len(testlabelnorm)))
	for i in xrange(len(testlabelnorm)):
		randval[i] = random.uniform(0,1)

	for i in xrange(len(testparamnorm[0])):
		permutinput = np.zeros(np.shape(testparamnorm))
		permutinput[:] = testparamnorm
		permutinput[:,i] = randval
		G3pred = mod.predict(permutinput, batch_size=batch_size)
		G3real = G3pred*20.0
		err = mean_squared_error(testlabel, G3real)
		permutsample[trying, i] = err

print permutsample
#print testparamnorm

#print mean and standard deviation of error
errperformance = np.zeros((len(testparamnorm[0]), 2))
for i in xrange(len(testparamnorm[0])):
	errperformance[i,0] = np.mean(permutsample[:,i])
	errperformance[i,1] = np.std(permutsample[:,i])
errperformance[:,0] = errreal - errperformance[:,0]

print errperformance
