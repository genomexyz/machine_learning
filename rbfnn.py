#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, LSTM, RepeatVector, TimeDistributed

#setting
data = '/home/genomexyz/Downloads/titanic/train-mod.csv'
kval = 10
itertot = 40
sigma = 1.2
itergd = 300

def transforminput(param, center):
	newinput = np.zeros((len(param), len(center))).astype('float32')
	for i in xrange(len(param)):
		for j in xrange(len(center)):
			newinput[i,j] = np.exp(-(np.sum((param[i] - center[j])**2.0)**0.5) / sigma**2.0)
	return newinput

def generatemodel(numparam):
	model = Sequential()
	model.add(Dense(1, input_dim=numparam, activation='sigmoid'))
#	model.add(Dense(10, activation='sigmoid'))
#	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

dataread = np.genfromtxt(data, delimiter=',')[1:,1:]

alldata = []
for i in xrange(len(dataread)):
	if np.isnan(dataread[i,-2]):
		continue
	alldata.append(dataread[i])

alldata = np.asarray(alldata)

#dividing data
trainparam = alldata[:600,1:]
trainlabel = alldata[:600,0]

testparam = alldata[600:,1:]
testlabel = alldata[600:,0]

###############
#normalization#
###############

std = np.zeros((len(trainparam[0]))).astype('float32')
rata = np.zeros((len(trainparam[0]))).astype('float32')
trainparamnorm = np.zeros(np.shape(trainparam))
testparamnorm = np.zeros(np.shape(testparam))
for i in xrange(len(trainparam[0])):
	std[i] = np.std(trainparam[:,i])
	rata[i] = np.mean(trainparam[:,i])
	trainparamnorm[:,i] = (trainparam[:,i] - rata[i]) / std[i]
	testparamnorm[:,i] = (testparam[:,i] - rata[i]) / std[i]

###############
#search k-mean#
###############

#init kmean
kmean = np.zeros((kval, len(trainparamnorm[0])))

for i in xrange(kval):
	for j in xrange(len(kmean[0])):
		kmean[i,j] = random.uniform(min(trainparamnorm[:,j]),max(trainparamnorm[:,j]))

#looping of real algorithm
distmin = np.zeros((len(trainparamnorm)))
for i in xrange(itertot):
	print 'iterasi ke', i
	for j in xrange(len(distmin)):
		#determine euclid distance
		distall = np.sum((trainparamnorm[j] - kmean)**2.0, axis=1)**0.5
		distmin[j] = np.argmin(distall)

	#search new k mean
	for j in xrange(kval):
		clust = []
		for k in xrange(len(distmin)):
			if distmin[k] == j:
				clust.append(trainparamnorm[k])
		if len(clust) > 0:
			kmean[j] = np.mean(np.asarray(clust), axis=0)

#tranform our input
newinput = transforminput(trainparamnorm, kmean)

print trainlabel
##########################
#gradient descent session#
##########################

mod = generatemodel(kval)
mod.fit(newinput, trainlabel, batch_size=20, epochs=itergd, verbose=1, shuffle=True)		

##################
#predict session#
##################

#transform test data
newinputtest = transforminput(testparamnorm, kmean)

lifeprob = mod.predict(newinputtest)

#######################
#determine performance#
#######################

#determine biner accuracy
binpred = np.zeros((len(lifeprob)))
for i in xrange(len(lifeprob)):
	if lifeprob[i] > 0.5:
		binpred[i] = 1.

score = 0
for i in xrange(len(testlabel)):
	if binpred[i] == testlabel[i]:
		score += 1
accbin = float(score) / float(len(testlabel))

#determine brier score
brierscore = 0
for i in xrange(len(testlabel)):
	brierscore += (testlabel[i] - lifeprob[i])**2.0
brierscore = brierscore / float(len(testlabel))

for i in xrange(len(testlabel)):
	print lifeprob[i], testlabel[i]
print accbin, brierscore[0]
