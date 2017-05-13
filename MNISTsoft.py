#!/usr/bin/python

#tensorflow
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

from mpl_toolkits.basemap import Basemap
from matplotlib import colors
from scipy import misc
import glob
import numpy as np
import csv
import matplotlib.pyplot as plt
import time

def readimage(numsam, iterlearn, dirname, howtoaccess):
	print('read image')
	print('wait...')
	rawdata = []
	for i in xrange(numsam):
		counter = 1
		for image_path in glob.glob(dirname+str(i)+howtoaccess):
			if counter > iterlearn:
				break
			image = misc.imread(image_path)
			rawdata.append(image)
			counter += 1
	print('done')
	return rawdata

#normalizing
def batnorm(matrix):
	rata = np.mean(matrix)
	variation = np.var(matrix)
	return (matrix - rata) / (variation)**0.5

#one hot for true class
def onehot(classtot, realclass):
	klass = np.zeros(classtot)
	klass[int(realclass)] = 1.0
	return klass

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#print(mnist[:])

#setting
samplenum = 10
iterperlearn = 200
dirnameacc = '/home/genomexyz/mnist_png/training/'
accessing = '/*.png'
learnrate = 0.8
iterasi = 1000
imgdim = 28
#regression = sx + c
#c_begin = 0
#s_begin = 0

#########################
#operation of tensorflow#
#########################

#at tensorflow, there is 2 variably type:
#- variable that contain value
#- variable as function

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

#define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

#cross entropy (function)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#training step (function)
train_step = tf.train.GradientDescentOptimizer(learnrate).minimize(cross_entropy)

#try to predict (function)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

#Softmax function
model = tf.nn.softmax(tf.matmul(x, W) + b)

#determine accuration
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#######################
#read raw data session#
#######################

rawdata = readimage(samplenum, iterperlearn, dirnameacc, accessing)

rawdataflat = np.zeros((samplenum*iterperlearn, imgdim*imgdim))
for i in xrange(len(rawdata)):
	rawdataflat[i] = rawdata[i].flatten()

#####################
#normalizing session#
#####################
for i in xrange(len(rawdataflat)):
	rawdataflat[i] = batnorm(rawdataflat[i])

##################
#training session#
##################

#create one hot true label
onetruelabel = np.zeros((samplenum*iterperlearn, samplenum))
for i in xrange(samplenum*iterperlearn):
	onetruelabel[i] = onehot(samplenum, i/iterperlearn)


with tf.Session() as sess:
	tf.global_variables_initializer().run()
	for i in range(iterasi):
		sess.run(train_step, feed_dict={x: rawdataflat, y_: onetruelabel})
	print(tf.argmax(y,1).eval(feed_dict={x: rawdataflat}))
	print(tf.argmax(y_,1).eval(feed_dict={y_: onetruelabel}))
	#weight = W.eval()
	#bias = b.eval()


####################
#predicting session#
####################

	#read raw
	testdata = readimage(samplenum, 200, dirnameacc, accessing)
	#flatten
	rawdataflattest = np.zeros((samplenum*200, imgdim*imgdim))
	for i in xrange(len(testdata)):
		rawdataflattest[i] = testdata[i].flatten()
	#normalizing
	for i in xrange(len(rawdataflat)):
		rawdataflattest[i] = batnorm(rawdataflat[i])
	onetruelabeltest = np.zeros((samplenum*200, samplenum))
	for i in xrange(samplenum*200):
		onetruelabeltest[i] = onehot(samplenum, i/200)
	print(sess.run(accuracy, feed_dict={x: rawdataflattest, y_: onetruelabeltest}))
	#print(tf.argmax(y,1).eval(feed_dict={x: rawdataflattest}))

#with tf.Session() as sess:
#	res = sess.run(multip)

