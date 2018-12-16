#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#setting
datasrc = 'datacluster.dat'
neighbor = 4.0
radius = 0.015

#read data
data = np.genfromtxt(datasrc, delimiter=[8, 8])

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

#extra 2 column for: class (index 0) and current state (index 1)
#state-> 0 = not scanned, 1 = scanned, 2 = checked
#class-> 0 = class unknown
unkdatastat = np.zeros((len(datanorm), 2))

################
#real algorithm#
################

running = True
clusternumbering = 0
while running:
	print 'number of cluster', clusternumbering
	hitung = 0
	for i in xrange(len(unkdatastat)):
		if unkdatastat[i,1] == 2:
			hitung += 1
	print hitung
	#scan all data, check if it already scanned or not
	#modescan = True, clustering with already known data otherwise construct new cluster
	modescan = False
	for i in xrange(len(unkdatastat)):
		if unkdatastat[i,1] == 1:
			datacheck = i
			modescan = True
			break
	if modescan:
		cntdist = []

		#assign data to cluster
		for i in xrange(len(datanorm)):
			#get all euclid distance to check neighbor
			if i == datacheck:
				continue
			dist = np.sum((datanorm[datacheck] - datanorm[i])**2.0)**0.5
			if dist < radius:
				cntdist.append(i)
		if len(cntdist) >= neighbor:
			for j in xrange(len(cntdist)):
				if unkdatastat[int(cntdist[j]), 1] != 2:
					unkdatastat[int(cntdist[j]), 0] = clusternumbering
					unkdatastat[int(cntdist[j]), 1] = 1
			unkdatastat[datacheck,0] = clusternumbering
			unkdatastat[datacheck,1] = 2
		else:
			unkdatastat[datacheck,1] = 2
	#construct new cluster
	else:
		cntdist = []
		running = False
		clusternumbering += 1
		#scan once again
		for i in xrange(len(unkdatastat)):
			if unkdatastat[i,1] == 0:
				datacheck = i
				running = True
				break

		if running:
			#assign data to cluster
			for i in xrange(len(datanorm)):
				#get all euclid distance to check neighbor
				if i == datacheck:
					continue
				dist = np.sum((datanorm[datacheck] - datanorm[i])**2.0)**0.5
				if dist < radius:
					cntdist.append(i)
			if len(cntdist) >= neighbor:
				for j in xrange(len(cntdist)):
					if unkdatastat[int(cntdist[j]), 1] != 2:
						unkdatastat[int(cntdist[j]), 0] = clusternumbering
						unkdatastat[int(cntdist[j]), 1] = 1
				unkdatastat[datacheck,0] = clusternumbering
				unkdatastat[datacheck,1] = 2
			else:
				unkdatastat[datacheck,1] = 2
				clusternumbering -= 1



##########
#plotting#
##########

#prepare color
colors = cm.hsv(np.linspace(0, 1, clusternumbering+1))

#plot the change of clustering
alllabelcol = []
for i in xrange(len(unkdatastat)):
	col = colors[int(unkdatastat[i,0])]
	alllabelcol.append(col)
alllabelcol = np.asarray(alllabelcol)

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(data[:,0],data[:,1], color=alllabelcol,s=5)

fig.savefig('clustering/DBSACN')
