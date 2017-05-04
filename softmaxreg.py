#!/usr/bin/python

from mpl_toolkits.basemap import Basemap
from matplotlib import colors
from scipy import misc
import glob
import numpy as np
import csv
import matplotlib.pyplot as plt
import random

def linear (slope, constant, x):
	""" compute linear function for softmax """
	return slope * x + constant

def pretrain(dimen, klas):
	pretraining = np.zeros(dimen)
	pretraining[klas,:,:] = 1
	return pretraining

def predicting(slope, bias, trypred):
	votemat = softmax(linear(slope, bias, trypred))
	for i in xrange(samplenum):
		print np.sum(votemat[i])

def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	if x.ndim == 1:
		sumsum = np.sum(np.exp(x))
		return np.exp(x) / sumsum
	elif x.ndim == 2:
		sumsum = np.zeros_like(x)
		M,N = x.shape
		for i in xrange (N):
			sumsum[:,i] = np.sum(np.exp(x[:,i]))
		return np.exp(x) / sumsum
	elif x.ndim == 3:
		sumsum = np.zeros_like(x)
		M,N,O = x.shape
		for i in xrange(N):
			for j in xrange(O):
				sumsum[:,i,j] = np.sum(np.exp(x[:,i,j]))
		#print sumsum[:,2,2]
		#print sumsum[:,19,19]
		#print x
		#print np.max(sumsum)
		return np.exp(x) / sumsum

#setting
samplenum = 9
iterperlearn = 20
dirnameacc = '/home/genomexyz/mnist_png/training/'
accessing = '/*.png'
learnrate = 0.5
iterasi = 100
#regression = sx + c
#c_begin = 0
#s_begin = 0

print 'read image'
print 'wait...'
rawdata = []
for i in xrange(samplenum):
	counter = 1
	for image_path in glob.glob(dirnameacc+str(i)+accessing):
		if counter > iterperlearn:
			break
		image = misc.imread(image_path)
		rawdata.append(image)
		counter += 1
print 'done'
#making 2 array of proces, 1 for slope array and 1 for constant array -> linear regression for softmax
#for every possibility of output (classification) have a dimension.
#So in our case the dimension is [9] for variable ->[28][28] pixel of pic
Y, X = np.shape(rawdata[0])
rawdata = np.asarray(rawdata)
dataprocslope = np.zeros([samplenum, Y, X])
dataproccons = np.zeros([samplenum, Y, X])


#for i in xrange(samplenum):
#	dataprocslope.append(np.zeros(np.shape(rawdata[0])))
#	dataproccons.append(np.zeros(np.shape(rawdata[0])))

#for i in xrab
#print np.shape(dataprocslope)
#print dataprocslope

#Y, X = np.shape(rawdata[0])
#for i in xrange(iterasi):
#	print 'iterasi ke', i
#	for row in xrange(Y):
#		for column in xrange(X):
#			s_grad = np.zeros(samplenum)
#			c_grad = np.zeros(samplenum)
#			for j in xrange(samplenum):
#				for k in xrange(iterperlearn):
#					#print np.shape(dataprocslope[:,0,0])
#					possible = softmax(linear(dataprocslope[:,row,column], dataproccons[:,row,column], \
#					rawdata[(j*iterperlearn)+k][row][column]))
					#print np.shape(possible)
					#print possible
#					for l in xrange(samplenum):
#						if l == j:
#							yesorno = 1.0
#						else:
#							yesorno = 0.0
#						s_grad[l] += rawdata[(j*iterperlearn)+k][row][column] * (yesorno - possible[l])
#						c_grad[l] += (yesorno - possible[l])
			#update coef and cons
#			s_grad = s_grad * -1.0 / (samplenum * iterperlearn)
#			c_grad = c_grad * -1.0 / (samplenum * iterperlearn)
#			dataprocslope[:,row,column] -= learnrate * s_grad
#			dataproccons[:,row,column] -= learnrate * c_grad
#			print dataprocslope[:,row,column]

#print np.sum(dataproccons)
#print 'coba bandingkan'
#print np.sum(dataprocslope)


#pretend this algorithm is true!
Y, X = np.shape(rawdata[0])
counter = 0
for looping in xrange(iterasi):
	print 'iterasi ke', looping
	for i in xrange(samplenum*iterperlearn):
		s_grad = np.zeros([samplenum, Y, X])
		c_grad = np.zeros([samplenum, Y, X])
		possible = softmax(linear(dataprocslope, dataproccons, rawdata[i]))
		yesorno = pretrain(np.shape(dataprocslope), i/iterperlearn)
		#print yesorno
		s_grad += (rawdata[i] * (yesorno - possible))
		#print np.sum(s_grad)
		c_grad += (yesorno - possible)
	#print np.sum(dataprocslope)
	s_grad = s_grad / (samplenum*iterperlearn)
	c_grad = c_grad / (samplenum*iterperlearn)
	dataprocslope += learnrate * s_grad
	dataproccons += learnrate * c_grad

print dataprocslope
print 'bandingkan'
print dataproccons
print 'untuk sum slope', np.sum(dataprocslope), 'dan ini untuk cons', np.sum(dataproccons)
print 'coba hitung'
print dataprocslope * 5 + dataproccons

print predicting(dataprocslope, dataproccons, rawdata[10])

#coba normalization pixel pada gambar
