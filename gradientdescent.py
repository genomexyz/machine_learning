#!/usr/bin/python

from mpl_toolkits.basemap import Basemap
from matplotlib import colors
import numpy as np
import csv
import matplotlib.pyplot as plt

#setting
learnrate = 0.0001
iterasi = 1000
#regression = sx + c
c_begin = 5
s_begin = 5

datafilename = 'datasampel'
#y array
analysis = []


with open(datafilename, 'r') as input_file:
	reader = csv.reader(input_file, delimiter = '\t')
	for line in reader:
		if line:	#check if list is not empty
			analysis.append(float(line[1]))
print analysis

#x array
week = range(1,len(analysis)+1)

#error array
err = []
for i in xrange(iterasi):
	s_grad = 0.0
	c_grad = 0.0
	errtemp = 0.0
	for j in xrange(len(week)):
		s_grad += (analysis[j] - (s_begin * week[j] + c_begin)) * week[j]
		c_grad += (analysis[j] - (s_begin * week[j] + c_begin))
	s_grad = -(2.0/len(week)) * s_grad
	c_grad = -(2.0/len(week)) * c_grad
	s_begin = s_begin - learnrate * s_grad
	c_begin = c_begin - learnrate * c_grad
	#calculate error
	for k in xrange(len(week)):
		errtemp += abs(analysis[k] - (s_begin * week[k] + c_begin))
	err.append(errtemp/len(week))

#this is the coefficient and contant we get
print c_begin, s_begin

ideal = []
for data in week:
	ideal.append(s_begin * data + c_begin)

print s_begin, c_begin
plt.plot(week,analysis, 'o')
plt.plot(week, ideal)
plt.ylabel('jumlah penjualan mobil')
plt.show()
plt.ylabel('rata-rata error absolute')
plt.xlabel('jumlah iterasi')
plt.title('perkembangan error ordinary gradient descent')
plt.plot(range(iterasi), err)
plt.show()
