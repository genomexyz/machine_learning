#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import os
import sys

#setting
traindata = '/home/genomexyz/risetCPNN/training data/train-6.csv'
testdata = '/home/genomexyz/risetCPNN/training data/test-6.csv'
sigma = 0.5
gamma = 0.8

#do automation
#arg -> program fase

traindata = '/home/genomexyz/risetCPNN/training data/train-'+sys.argv[1]+'.csv'
testdata = '/home/genomexyz/risetCPNN/training data/test-'+sys.argv[1]+'.csv'

##############
#prepare data#
##############
 
traindatamat = np.genfromtxt(traindata,delimiter=',')[1:]
testdatamat = np.genfromtxt(testdata,delimiter=',')[1:]

trainparam = traindatamat[:,:-1]
trainlabel = traindatamat[:,-1]
testparam = testdatamat[:,:-1]
testlabel = testdatamat[:,-1]

#save rain data
hujan = np.zeros(len(testlabel))
hujan[:] = testlabel[:]

#calc rain and no rain class
raintot = 0
noraintot = 0
rainclassparam = []
norainclassparam = []
for i in xrange(len(trainlabel)):
	if trainlabel[i] > 0.0:
		raintot += 1
		rainclassparam.append(trainparam[i])
	else:
		noraintot += 1
		norainclassparam.append(trainparam[i])

#print raintot, noraintot
rainclassparam = np.asarray(rainclassparam)
norainclassparam = np.asarray(norainclassparam)

###############
#normalization#
###############
#get mean and std (train)
std = np.zeros((len(trainparam[0]))).astype('float32')
rata = np.zeros((len(trainparam[0]))).astype('float32')
for i in xrange(len(trainparam[0])):
	std[i] = np.std(trainparam[:,i])
	rata[i] = np.mean(trainparam[:,i])
	rainclassparam[:,i] = (rainclassparam[:,i] - rata[i]) / std[i]
	norainclassparam[:,i] = (norainclassparam[:,i] - rata[i]) / std[i]
	testparam[:,i] = (testparam[:,i] - rata[i]) / std[i]

################
#real algorithm#
################

allgausspos = np.zeros((len(testparam), len(rainclassparam)))
allgaussneg = np.zeros((len(testparam), len(norainclassparam)))

totvar = float(len(testparam[0]))

#print np.shape(rainclassparam), np.shape(norainclassparam)

#calculate gauss for positive class
for i in xrange(len(testparam)):
	allgausspos[i] = 1.0 / ((2*np.pi)**(totvar/2.0)*sigma**totvar) * np.exp(-(\
	np.sum((testparam[i,:] - rainclassparam)**2.0, axis=1)**0.5 / (2.0*sigma**2.0)))
	#print (testparam[i,:] - norainclassparam)**2.0

#calculate gauss for negative class
for i in xrange(len(testparam)):
	allgaussneg[i] = 1.0 / ((2*np.pi)**(totvar/2.0)*sigma**totvar) * np.exp(-(\
	np.sum((testparam[i,:] - norainclassparam)**2.0, axis=1)**0.5 / (2.0*sigma**2.0)))
	#print (testparam[i,:] - norainclassparam)**2.0

#sort allgauss
allgausspos = np.fliplr(np.sort(allgausspos))[:,:int(gamma*len(allgausspos[0]))]
allgaussneg = np.fliplr(np.sort(allgaussneg))[:,:int(gamma*len(allgaussneg[0]))]

#print 'lihat dimensi', len(allgaussneg[0]), len(allgausspos[0])

#calculate conditional probability
hujanprob = np.zeros((len(testparam)))
for i in xrange(len(hujanprob)):
	hujanprob[i] = np.mean(allgausspos[i]) / (np.mean(allgausspos[i]) + np.mean(allgaussneg[i]))

#print hujanprob
#np.savetxt('risetCPNN/hasil.csv', hujanprob, delimiter=',')

######################
#calculate accuration#
######################

#convert test label from cont value to binary (rain or not)
for i in xrange(len(testlabel)):
	if testlabel[i] > 0.0:
		testlabel[i] = 1.
	else:
		testlabel[i] = 0.

#determine biner accuracy
binpred = np.zeros((len(hujanprob)))
for i in xrange(len(hujanprob)):
	if hujanprob[i] > 0.5:
		binpred[i] = 1.

score = 0
for i in xrange(len(testlabel)):
	#print 'coba cek ini', binpred[i], testlabel[i], hujanprob[i], hujan[i]
	if binpred[i] == testlabel[i]:
		score += 1
accbin = float(score) / float(len(testlabel))

#determine brier score
brierscore = 0
for i in xrange(len(testlabel)):
	brierscore += (testlabel[i] - hujanprob[i])**2.0
	#if testlabel[i] == 1.:
	#	brierscore += (hujanprob[i])**2.0
	#else:
	#	brierscore += (1. - hujanprob[i])**2.0
brierscore = brierscore / float(len(testlabel))

hit = 0.
fa = 0.
miss = 0.
cn = 0.
for i in xrange(len(testlabel)):
	if (binpred[i] == 1.0) and (testlabel[i] == 1.0):
		hit += 1.
	elif (binpred[i] == 1.0) and (testlabel[i] == 0.0):
		fa += 1.
	elif (binpred[i] == 0.0) and (testlabel[i] == 1.0):
		miss += 1.
	elif (binpred[i] == 0.0) and (testlabel[i] == 0.0):
		cn += 1.

CSI = hit / (fa+miss)
POD = hit / (hit+miss)
FAR = fa / (fa+hit)
#print 'hit, false alarm, miss, and correct negative'
#print hit, fa, miss, cn
#print 'CSI', CSI
#print 'POD', POD
#print 'FAR', FAR
#print 'akurasi', accbin
#print 'brier scorenya', brierscore
#print akurasi	POD	CSI	FAR	brier score
#print accbin, POD, CSI, FAR, brierscore
print brierscore
