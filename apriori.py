#!/usr/bin/python

import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
import sys

#himpunan as tested set
#peop is array of reviewer
#dictset is dictionary with reviewer as key and movieid as value
def calcfreq(himpunan, peop, dictset):
	cnt = 0
	for satuan in peop:
		if himpunan.issubset(dictset[satuan]):
			cnt += 1
	return cnt
		

minfreq = 1000
maxset = 5
totnumtoshow = 2
nomorid = 1

#retrieve argument
if len(sys.argv) == 2:
	nomorid = int(sys.argv[1])
	print 'mencari relasi 2 id dengan id {0}'.format(nomorid)
elif len(sys.argv) >= 3:
	nomorid =  int(sys.argv[1])
	totnumtoshow = int(sys.argv[2])
	print 'mencari relasi {0} id dengan id {1}'.format(totnumtoshow, nomorid)
else:
	print 'menggunakan nilai default -> relasi 2 id dengan id 1'

ratingfile = '/home/genomexyz/ml-1m/ratings.dat'
rate = pd.read_csv(ratingfile, delimiter='::', header=None, \
names = ["UserID", "MovieID", "Rating", "Datetime"], engine = 'python')
rate['favorit'] = rate['Rating'] > 3

	
#tes = pd.DataFrame([[1,2,3], [1,2,3]]) -> there is 3 column and 2 row


#make init set of people and their review
people = np.unique(rate['UserID'].values)
review = [{0} for i in xrange(len(people))]

#make dict of people and their favorit movie
pepreview = dict(zip(people,review))
for row in rate.values:
	if row[4]:
		pepreview[row[0]].add(row[1])

#make first iter
word = np.unique(rate['MovieID'].values)
arti = np.zeros(len(word))
kamus = dict(zip(word,arti))
for row in rate.values:
	if row[4]:
		kamus[row[1]] += 1
#make important table
ratepd = pd.DataFrame(kamus.items(), columns = ['movieid', 'favorit']).sort_values(by = 'favorit', ascending = False)

#elminate by minimum frequency
resfirst = [row for row in ratepd.values if row[1] > minfreq]
iteritem = np.zeros(len(resfirst), dtype = np.int)
valfreqfrst = []
for i in xrange(len(resfirst)):
	iteritem[i] = resfirst[i][0]
	valfreqfrst.append(int(resfirst[i][1]))

#store itemset in dictionary
kitem = defaultdict(int)
kitem[1] = zip(iteritem, valfreqfrst)


#make second iter etc
for i in xrange(2, totnumtoshow+1):
	#making list of combination from iter1
	combigen = list(combinations(iteritem, i))
	#container for combination that pass a minfreq test
	container = []
	for j in xrange(len(combigen)):
		freq = calcfreq(set(combigen[j]), people, pepreview)
		if freq > minfreq:
			container.append((combigen[j], freq))
	kitem[i] = container

#######################################################
#FINAL STEP -> CALCULATE THE CONFIDENCE FOR ALL MEMBER#
#######################################################

#confidence = sum(A & B) / sum(A)

maxfreq = 0
for i in xrange(len(kitem[totnumtoshow])):
	#print {nomorid}.issubset(set([9,4,5]))
	cont = set(kitem[totnumtoshow][i][0])
	if not {nomorid}.issubset(set(kitem[totnumtoshow][i][0])):
		valfreq = kitem[totnumtoshow][i][1]
		valcom = kitem[totnumtoshow][i][0]
		newcombi = set(kitem[totnumtoshow][i][0])
		newcombi.add(nomorid)
		newcombifreq = calcfreq(newcombi, people, pepreview) / float(valfreq)
		if newcombifreq > maxfreq:
			maxfreq = newcombifreq
			maxid = valcom
#print 'jadi kesimpulannya orang yang menyukai '+str(maxid[0])+' dan '+str(maxid[1])+ \
#' juga menyukai '+str(nomorid)+' dengan nilai confidence '+str(maxfreq)
#print "this is a tuple: %s" % (thetuple,)
print 'jadi kesimpulannya orang yang menyukai {0} juga menyukai {1} dengan nilai confidence {2}%'.format(maxid,nomorid,(maxfreq * 100))

#ratepd = pd.DataFrame({"key": d.keys(), "value": d.values()})
#print ratepd.sort('favorit', ascending = False)
