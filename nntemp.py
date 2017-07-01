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
errbound = 0.55
learning_rate = 0.05
iterasi = 7000
totvar = 5
hidnode = 5
datatrain = 'datann'
datatest = 'testemp'


def batnorm(matrix):
	rata = np.mean(matrix)
	variation = np.std(matrix)
	return (matrix - rata) / (variation)

def batnorm2(matrix):
	return (matrix - matrix.min()) / (matrix.max() - matrix.min())

#######################
#read raw data session#
#######################

#
#read data test
#
wndspdx = np.asarray([])
pressurex = np.asarray([])
drybulbx = np.asarray([])
RHx = np.asarray([])
wetbulbx = np.asarray([])
with open(datatest, 'r') as input_file:
	reader = csv.reader(input_file, delimiter = '\t')
	for line in reader:
		#all line that not empty and not start with '#' (comment section)
		if line and not line[0].startswith('#'):
			wndspdx = np.append(wndspdx, float(line[0]))
			pressurex = np.append(pressurex, 1000 + float(line[5]) / 10.0)
			drybulbx = np.append(drybulbx, float(line[6]) / 10.0)
			RHx = np.append(RHx, float(line[-3]))
			wetbulbx = np.append(wetbulbx, float(line[-1]) / 10.0)

#test data (persistence method)
perstrue = np.delete(drybulbx, 0, 0)
perspred = np.delete(drybulbx, -1, 0)

#true data
drybulbtruex = np.zeros((len(drybulbx)-1, 1))
drybulbtruex[:,0] = np.delete(drybulbx, 0, 0)

wndspdx = np.delete(wndspdx, -1, 0)
pressurex = np.delete(pressurex, -1, 0)
drybulbx = np.delete(drybulbx, -1, 0)
RHx = np.delete(RHx, -1, 0)
wetbulbx = np.delete(wetbulbx, -1, 0)

inputcompx = np.zeros((len(wndspdx), totvar))
inputcompx[:,0] = wndspdx
inputcompx[:,1] = pressurex
inputcompx[:,2] = drybulbx
inputcompx[:,3] = RHx
inputcompx[:,4] = wetbulbx



#
#read data training
#
wndspd = np.asarray([])
pressure = np.asarray([])
drybulb = np.asarray([])
RH = np.asarray([])
wetbulb = np.asarray([])
with open(datatrain, 'r') as input_file:
	reader = csv.reader(input_file, delimiter = '\t')
	for line in reader:
		#all line that not empty and not start with '#' (comment section)
		if line and not line[0].startswith('#'):
			wndspd = np.append(wndspd, float(line[0]))
			pressure = np.append(pressure, 1000 + float(line[5]) / 10.0)
			drybulb = np.append(drybulb, float(line[6]) / 10.0)
			RH = np.append(RH, float(line[-3]))
			wetbulb = np.append(wetbulb, float(line[-1]) / 10.0)

#true data
drybulbtrue = np.zeros((len(drybulb)-1, 1))
drybulbtrue[:,0] = np.delete(drybulb, 0, 0)
truthdrybulb = np.delete(drybulb, 0, 0)
persistdrybulb = np.delete(drybulb, -1, 0)
meandrybulbtrue = np.mean(drybulbtrue[:,0])
stddrybulbtrue = np.std(drybulbtrue[:,0])

wndspd = np.delete(wndspd, -1, 0)
pressure = np.delete(pressure, -1, 0)
drybulb = np.delete(drybulb, -1, 0)
RH = np.delete(RH, -1, 0)
wetbulb = np.delete(wetbulb, -1, 0)

inputcomp = np.zeros((len(wndspd), totvar))
inputcomp[:,0] = wndspd
inputcomp[:,1] = pressure
inputcomp[:,2] = drybulb
inputcomp[:,3] = RH
inputcomp[:,4] = wetbulb


#####################
#normalizing session#
#####################

#normalizing data training
contmean = np.zeros(totvar)
contstd = np.zeros(totvar)
for i in xrange(totvar):
	contmean[i] = np.mean(inputcomp[:,i])
	contstd[i] = np.std(inputcomp[:,i])
	inputcomp[:,i] = (inputcomp[:,i] - contmean[i]) / contstd[i]

#normalizing truth data
drybulbtrue[:,0] = batnorm(drybulbtrue[:,0])

#normalizing data test
for i in xrange(totvar):
	inputcompx[:,i] = (inputcompx[:,i] - contmean[i]) / contstd[i]

#########################
#operation of tensorflow#
#########################

#input layer
x = tf.placeholder(tf.float32, [None, totvar])
W = tf.Variable(tf.random_normal([totvar, hidnode]))
b = tf.Variable(tf.random_normal([hidnode]))
regin = tf.matmul(x, W) + b

#hidden layer 1st
xh = tf.sigmoid(regin)
Wh = tf.Variable(tf.random_normal([hidnode, hidnode]))
bh = tf.Variable(tf.random_normal([hidnode]))
finpred = tf.matmul(xh, Wh) + bh

#hidden layer 2nd
xhh = tf.sigmoid(finpred)
Whh = tf.Variable(tf.random_normal([hidnode, 1]))
bhh = tf.Variable(tf.random_normal([1]))
finpredh = tf.matmul(xhh, Whh) + bhh

#label
Y = tf.placeholder("float")

#loss function
cost = tf.reduce_mean(tf.pow(finpredh-Y, 2))
#lossfunc = tf.reduce_mean(tf.abs(pred-Y))

#optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


##################
#training session#
##################

deter = np.zeros((iterasi))
with tf.Session() as sess:
	tf.global_variables_initializer().run()
	for i in range(iterasi):
		sess.run(optimizer, feed_dict={x: inputcomp, Y: drybulbtrue})
		pre = finpredh.eval(feed_dict={x: inputcomp})[:,0]
		deter[i] = np.sqrt((np.mean((pre * stddrybulbtrue + meandrybulbtrue - truthdrybulb)**2.0)))
		#print(lossfunc.eval(feed_dict={x: inputcomp, Y: drybulbtrue}))
		#print(W.eval())
	hidw = Wh.eval()
	prediksi = finpredh.eval(feed_dict={x: inputcompx})[:,0]
	#akurasi = np.sqrt(np.mean((prediksi * stddrybulbtrue + meandrybulbtrue - truthdrybulb)**2.0))
	akurasi = np.sqrt(np.mean((prediksi * stddrybulbtrue + meandrybulbtrue - perstrue)**2.0))
	#print(np.transpose(pred.eval(feed_dict={x: inputcomp})))
	#print(np.transpose(Y.eval(feed_dict={Y: drybulbtrue})))
	#print(W.eval())

#deter = np.asarray([])
#with tf.Session() as sess:
#	tf.global_variables_initializer().run()
#	cnt = 0
#	while(True):
#		cnt += 1
#		sess.run(optimizer, feed_dict={x: inputcomp, Y: drybulbtrue})
#		pre = finpred.eval(feed_dict={x: inputcomp})[:,0]
#		detererr = np.sqrt((np.mean((pre * stddrybulbtrue + meandrybulbtrue - truthdrybulb)**2.0)))
#		deter = np.append(deter, detererr)
#		if (detererr < errbound):
#			break
#	hidw = Wh.eval()
#	prediksi = finpred.eval(feed_dict={x: inputcomp})[:,0]
#	akurasi = np.sqrt(np.mean((prediksi * stddrybulbtrue + meandrybulbtrue - truthdrybulb)**2.0))

#REMINDER!
#OUTPUT MUST BE 2-DIMENSION MATRIX

#################
#predict session#
#################

print(prediksi * stddrybulbtrue + meandrybulbtrue)
print(perstrue)
#print(truthdrybulb)
print('error terhadap data training ', deter[iterasi-1])
print('error yang sebenarnya', akurasi)
print('error metode persistensi', np.sqrt((np.mean((perspred - perstrue)**2.0))))
#print(np.sqrt(np.mean((persistdrybulb - truthdrybulb)**2.0)))
#print('iterasi that needed ',cnt)

##################
#plotting session#
##################

plt.ylabel('error')
plt.xlabel('iterasi')
plt.plot(range(iterasi), deter)
plt.show()

#plt.ylabel('error')
#plt.xlabel('iterasi')
#plt.plot(range(cnt), deter)
#plt.show()

#plt.ylabel('prediksi dry bulb')
#plt.xlabel('dry bulb')
#plt.plot(truthdrybulb, prediksi * stddrybulbtrue + meandrybulbtrue)
#plt.show()

#0.816800940422
