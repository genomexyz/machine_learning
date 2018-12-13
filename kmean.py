#!/usr/bin/python

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#setting
datasrc = 'datacluster.dat'
kval = 35
itertot = 50

def eudistance(vec1, vec2):
	return np.sum((vec1 - vec2)**2.0)**0.5

#read data
data = np.genfromtxt(datasrc, delimiter=[8, 8])

#prepare color
colors = cm.hsv(np.linspace(0, 1, kval))

###############
#normalization#
###############

maxcnt = np.zeros((len(data[0])))
mincnt = np.zeros((len(data[0])))
datanorm = np.zeros(np.shape(data))
for i in xrange(len(maxcnt)):
	maxcnt[i] = np.max(data[:,i])
	mincnt[i] = np.min(data[:,i])
	datanorm[:,i] = (data[:,i] - mincnt[i]) / (maxcnt[i] - mincnt[i])


################
#real algorithm#
################

#init kmean
kmean = np.zeros((kval, len(data[0])))
kmeanreal = np.zeros((kval, len(data[0])))

for i in xrange(kval):
	for j in xrange(len(data[0])):
		kmean[i,j] = random.uniform(0,1)

#looping of real algorithm
distmin = np.zeros((len(datanorm)))
for i in xrange(itertot):
	print 'iterasi ke', i
	for j in xrange(len(distmin)):
		#determine euclid distance
		distall = np.sum((datanorm[j] - kmean)**2.0, axis=1)**0.5
		distmin[j] = np.argmin(distall)

	#search new k mean
	for j in xrange(kval):
		clust = []
		for k in xrange(len(distmin)):
			if distmin[k] == j:
				clust.append(datanorm[k])
		if len(clust) > 0:
			kmean[j] = np.mean(np.asarray(clust), axis=0)
	
	#plot the change of clustering
	alllabelcol = []
	for j in xrange(len(distmin)):
		col = colors[int(distmin[j])]
		alllabelcol.append(col)
	alllabelcol = np.asarray(alllabelcol)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	scatter = ax.scatter(data[:,0],data[:,1], color=alllabelcol,s=5)

	#back to real scale
	for j in xrange(len(maxcnt)):
		kmeanreal[:,j] = kmean[:,j] * (maxcnt[j] - mincnt[j]) + mincnt[j]

	for j,k in kmeanreal:
		ax.scatter(j,k,s=10,c='black',marker='+')
	fig.suptitle('iterasi ke-'+str(i+1))
	fig.savefig('clustering/iter-'+str(i))


#############################
#save data in cluster format#
#############################

for i in xrange(len(distmin)):
	#determine euclid distance
	distall = np.sum((datanorm[i] - kmean)**2.0, axis=1)**2.0
	distmin[i] = np.argmin(distall)
allclust = []
for j in xrange(kval):
	clust = []
	for k in xrange(len(distmin)):
		if distmin[k] == j:
			clust.append(data[k])
	allclust.append(clust)

clustfile = open('clustering_data.dat', 'w')

for j in xrange(kval):
	clustfile.write('data in cluster '+str(j+1)+'\n')
	clustfile.write('--------------------------------\n')
	for k in xrange(len(allclust[j])):
		clustfile.write(str(allclust[j][k][0])+'\t'+str(allclust[j][k][1])+'\n')
	clustfile.write('\n\n')

