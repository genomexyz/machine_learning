#!/usr/bin/python

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.sequence import pad_sequences
import os
from PIL import Image

#setting
numclass = 4
dimx = 28
dimy = 28
data0 = '/home/genomexyz/mnist_png/training/0'
data1 = '/home/genomexyz/mnist_png/training/1'
data2 = '/home/genomexyz/mnist_png/training/2'
data3 = '/home/genomexyz/mnist_png/training/3'
thresread = 500

def separatedata(mat):
	trainmat = mat[0:400]
	testmat = mat[400:420]
	return trainmat, testmat

def generatemodel(ydim, xdim):
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(numclass, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def readimage(addr):
	matimage = []
	iterimage = sorted(os.listdir(addr))
	for i in xrange(1, len(iterimage)):
		if (i > thresread):
			break
		img = Image.open(addr+'/'+iterimage[i])
		imgarray = np.array(img)
		#print np.shape(imgarray)
		matimage.append(imgarray)
	matimage = np.asarray(matimage)
	return matimage

#get matrix of image
mat0 = readimage(data0)
mat1 = readimage(data1)
mat2 = readimage(data2)
mat3 = readimage(data3)

#separate matrix
train0 ,test0 = separatedata(mat0)
train1 ,test1 = separatedata(mat1)
train2 ,test2 = separatedata(mat2)
train3 ,test3 = separatedata(mat3)

#combine all train data and test data
traindata = np.concatenate((train0, train1, train2, train3))
testdata = np.concatenate((test0, test1, test2, test3))

#reshape train and test data
#VERY IMPORTANT: MAKE SURE TYPE OF DATA IS float32
traindata = np.reshape(traindata, (np.shape(traindata)[0], np.shape(traindata)[1], np.shape(traindata)[2], 1)).astype('float32')
testdata = np.reshape(testdata, (np.shape(testdata)[0], np.shape(testdata)[1], np.shape(testdata)[2], 1)).astype('float32')

#create on hot vector
trainlabel0 = np.zeros((len(train0), numclass))
trainlabel0[:,0] = 1
testlabel0 = np.zeros((len(test0), numclass))
testlabel0[:,0] = 1

trainlabel1 = np.zeros((len(train1), numclass))
trainlabel1[:,1] = 1
testlabel1 = np.zeros((len(test1), numclass))
testlabel1[:,1] = 1

trainlabel2 = np.zeros((len(train2), numclass))
trainlabel2[:,2] = 1
testlabel2 = np.zeros((len(test2), numclass))
testlabel2[:,2] = 1

trainlabel3 = np.zeros((len(train3), numclass))
trainlabel3[:,3] = 1
testlabel3 = np.zeros((len(test3), numclass))
testlabel3[:,3] = 1

trainlabel = np.concatenate((trainlabel0, trainlabel1, trainlabel2, trainlabel3))
testlabel = np.concatenate((testlabel0, testlabel1, testlabel2, testlabel3))

print np.shape(trainlabel)
print np.shape(testlabel)

#####################
#normalizing session#
#####################


for i in xrange(len(traindata)):
	traindata[i] = traindata[i] / 255.0
for i in xrange(len(testdata)):
	testdata[i] = testdata[i] / 255.0

#print np.mean(traindata)

#################
#convnet process#
#################

mod = generatemodel(dimy, dimx)
mod.fit(traindata, trainlabel, epochs=100, verbose=1, batch_size=len(traindata))

####################
#evaluation process#
####################

# Final evaluation of the model
scores = mod.evaluate(testdata, testlabel, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
print scores
