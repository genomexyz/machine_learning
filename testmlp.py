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

#setting
learning_rate = 0.05
iterasi = 17000
totvar = 3
hidnode = 5

def batnorm2(matrix):
	return np.mean(matrix), np.std(matrix)

def batnorm(matrix):
	rata = np.mean(matrix)
	variation = np.var(matrix)
	#return (matrix - rata) / (variation)**0.5
	return (matrix - rata) / np.std(matrix)

####################
#data input session#
####################

#test data
#sintes = np.sin(54)
#costes = np.cos(54)
#tantes = np.tan(54)


#truth
tru = np.zeros((69, 1))
tru[:,0] = range(1,70)
simpantru = range(1,70)

#normalizing truth data
tru[:,0] = batnorm(tru[:,0])

#param
sinus = np.sin(tru[:,0])
cosinus = np.cos(tru[:,0])
tangent = np.tan(tru[:,0])


inputcomp = np.zeros((len(tru), totvar))
for i in xrange(len(tru)):
	inputcomp[i,0] = sinus[i]
	inputcomp[i,1] = cosinus[i]
	inputcomp[i,2] = tangent[i]

testcomp = np.zeros((len(sintes), totvar))
for i in xrange(len(sintes)):
	testcomp[i,0] = sintes[i]
	testcomp[i,1] = costes[i]
	testcomp[i,2] = tantes[i]

#####################
#normalizing session#
#####################
#print(inputcomp)

#normalizing training data
contmean = np.zeros(totvar)
contstd = np.zeros(totvar)
for i in xrange(totvar):
	contmean[i], contstd[i] = batnorm2(inputcomp[:,i])
	inputcomp[:,i] = (inputcomp[:,i] - contmean[i]) / contstd[i]

#for i in xrange(totvar):
#	inputcomp[:,i] = batnorm(inputcomp[:,i])

#normalizing test data
#for i in xrange(totvar):
#	testcomp[:,i] = batnorm(testcomp[:,i])
#	testcomp[:,i] = (testcomp[:,i] - contmean[i]) / contstd[i]

#print(testcomp)
#########################
#operation of tensorflow#
#########################

#bias
#b = tf.Variable(tf.zeros([2]))

#input layer
X = tf.placeholder(tf.float32, [None, totvar])
#W = tf.Variable(tf.zeros([totvar, hidnode]))
W = tf.Variable(tf.random_normal([totvar, hidnode]))
#b = tf.Variable(tf.zeros([hidnode]))
b = tf.Variable(tf.random_normal([hidnode]))
pred = tf.matmul(X, W) + b

#hidden layer
sigm = tf.sigmoid(pred)
#Wh = tf.Variable(tf.zeros([hidnode, 1]))
Wh = tf.Variable(tf.random_normal([hidnode, 1]))
#bh = tf.Variable(tf.zeros([1]))
bh = tf.Variable(tf.random_normal([1]))
finpred = tf.matmul(sigm, Wh) + bh

#hidden layer second
#sigmx = tf.sigmoid(finpred)
#Whx = tf.Variable(tf.random_normal([hidnode, 1]))
#bhx = tf.Variable(tf.random_normal([1]))
#finpredx = tf.matmul(sigm, Whx) + bhx

#true val
Y = tf.placeholder(tf.float32, [None,1])

# Mean squared error
cost = tf.reduce_mean(tf.pow(finpred-Y, 2))

#minimize our lost function
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

##################
#training session#
##################

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	for i in range(iterasi):
		sess.run(optimizer, feed_dict={X: inputcomp, Y: tru})
		#print(cost.eval(feed_dict={X: inputcomp, Y: tru}))
	prediksi = finpred.eval(feed_dict={X: inputcomp})[:,0]
	second = sigm.eval(feed_dict={X: inputcomp})
	wei = Wh.eval()

#################
#predict session#
#################
	#prediksitest = finpred.eval(feed_dict={X: testcomp})[:,0]

print(prediksi * np.var(simpantru)**0.5 + np.mean(simpantru))
#print(prediksitest * np.var(simpantru)**0.5 + np.mean(simpantru))
print(np.var(simpantru)**0.5)
