#!/usr/bin/python

from mpl_toolkits.basemap import Basemap
from matplotlib import colors
import numpy as np
import csv
import matplotlib.pyplot as plt
import random

def sigmoid(slope, constant, varx):
	return 1.0 / (1.0 + np.exp(-1.0 * (slope * varx + constant)))

#setting
learnrate = 0.05
iterasi = 1000
#regression = sx + c
c_begin = 0
s_begin = 0

datafilename = 'student'
#y array
analysisfeat = []
analysislabel = []


with open(datafilename, 'r') as input_file:
	reader = csv.reader(input_file, delimiter = '\t')
	for line in reader:
		if line:	#check if list is not empty
			analysisfeat.append(float(line[0]))
			analysislabel.append(float(line[1]))
#print analysisfeat
#print
#print analysislabel
analysis = zip(analysisfeat, analysislabel)

########
#action#
########
#error array
err = []
for i in xrange(iterasi):
	s_grad = 0.0
	c_grad = 0.0
	errtemp = 0.0
	#xi, yi = random.choice(analysis)
	#s_grad = (yi -  sigmoid(s_begin, c_begin, xi)) * xi
	#c_grad = yi -  sigmoid(s_begin, c_begin, xi)
	for j in xrange(len(analysis)):
		s_grad += (analysislabel[j] -  sigmoid(s_begin, c_begin, analysisfeat[j])) * analysisfeat[j]
		c_grad += analysislabel[j] - sigmoid(s_begin, c_begin, analysisfeat[j])
	s_grad = s_grad / len(analysis)
	c_grad = c_grad / len(analysis)
	#determine error
	for j in xrange(len(analysis)):
		errtemp += (analysislabel[j] * np.log(sigmoid(s_begin, c_begin, analysisfeat[j])) + (1.0 - analysislabel[j]) * \
		np.log(1 - sigmoid(s_begin, c_begin, analysisfeat[j])))
	err.append(errtemp/len(analysis))
	s_begin = s_begin + learnrate * s_grad
	c_begin = c_begin + learnrate * c_grad

hasil = []
for i in xrange(len(analysis)):
	hasil.append(sigmoid(s_begin, c_begin, analysisfeat[i]))
	print hasil[i], analysislabel[i], analysisfeat[i]
print 'bandingkan'
print s_begin, c_begin

predict = []
for i in xrange(len(analysis)):
	if sigmoid(s_begin, c_begin, analysisfeat[i]) > 0.5:
		predict.append(1)
	else:
		predict.append(0)

print predict

#plot real data
plt.scatter(analysisfeat, analysislabel)
plt.show()

#plot prediction
plt.scatter(analysisfeat, predict)
plt.show()

plt.ylabel('log likelihood')
plt.xlabel('iterasi')
plt.title('tingkat keyakinan')
plt.plot(range(iterasi), err)
plt.show()
