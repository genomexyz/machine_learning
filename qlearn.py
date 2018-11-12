#!/usr/bin/python

import numpy as np
import random

#setting
row = 10
column = 10
iteration = 100
targetsym = 9
trapsym = 1
playersym = 7
lrate = 0.2
discount = 0.9
epsilon = 1.0
minepsilon = 0.01
maxstep = 100
decay_epsilon = 0.005
actchoice = ['bawah', 'atas', 'kanan', 'kiri']
#target
target = np.asarray([9,6])

trap = []
trap.append([7,6])
trap.append([7,4])
trap.append([9,0])
trap.append([8,6])
trap = np.asarray(trap)

def init_board(player):
	#make a board
	board = np.zeros((row, column))

	#init board
	board[target[0], target[1]] = targetsym
	for j in xrange(len(trap)):
		board[trap[j,0], trap[j,1]] = trapsym
	board[player[0], player[1]] = playersym
	return board

def calcreward(posplayer1, posplayer2, postarget):
	jarak1 = np.absolute(posplayer1-postarget)
	jarak2 = np.absolute(posplayer2-postarget)

	trapped = False
	#check if trapped
	for i in xrange(len(trap)):
		if np.array_equal(posplayer2, trap[i]):
			trapped = True
	if np.sum(jarak1-jarak2) > 0:
		reward = 1.
	else:
		reward = -1.
	if np.array_equal(posplayer2, target):
		reward = 100.
	elif trapped:
		reward = -100.
	return reward

def action(qtable, posplayer):
	global epsilon
	chance = random.uniform(0,1)
	if chance < epsilon:
		while True:
			choice = random.randrange(len(actchoice))
			if actchoice[choice] == 'bawah':
				cnt = posplayer[0]
				cnt += 1
				if cnt < row:
					posplayer[0] = cnt
					break
			elif actchoice[choice] == 'atas':
				cnt = posplayer[0]
				cnt -= 1
				if cnt > 0:
					posplayer[0] = cnt
					break
			elif actchoice[choice] == 'kanan':
				cnt = posplayer[1]
				cnt += 1
				if cnt < column:
					posplayer[1] = cnt
					break
			elif actchoice[choice] == 'kiri':
				cnt = posplayer[1]
				cnt -= 1
				if cnt > 0:
					posplayer[1] = cnt
					break
	else:
		choice = np.argmax(qtable[posplayer[0], posplayer[1]])
		if actchoice[choice] == 'bawah':
			cnt = posplayer[0]
			cnt += 1
			if cnt < row:
				posplayer[0] = cnt
		elif actchoice[choice] == 'atas':
			cnt = posplayer[0]
			cnt -= 1
			if cnt > 0:
				posplayer[0] = cnt
		elif actchoice[choice] == 'kanan':
			cnt = posplayer[1]
			cnt += 1
			if cnt < column:
				posplayer[1] = cnt
		elif actchoice[choice] == 'kiri':
			cnt = posplayer[1]
			cnt -= 1
			if cnt > 0:
				posplayer[1] = cnt
	#decaying epsilon
	epsilon -= decay_epsilon
	return choice, posplayer

def realaction(qtable, posplayer):
	choice = np.argmax(qtable[posplayer[0], posplayer[1]])
	if actchoice[choice] == 'bawah':
		cnt = posplayer[0]
		cnt += 1
		if cnt < row:
			posplayer[0] = cnt
	elif actchoice[choice] == 'atas':
		cnt = posplayer[0]
		cnt -= 1
		if cnt > 0:
			posplayer[0] = cnt
	elif actchoice[choice] == 'kanan':
		cnt = posplayer[1]
		cnt += 1
		if cnt < column:
			posplayer[1] = cnt
	elif actchoice[choice] == 'kiri':
		cnt = posplayer[1]
		cnt -= 1
		if cnt > 0:
			posplayer[1] = cnt
	return actchoice[choice], posplayer

############
#init board#
############

#make q table
qtab = np.zeros((row, column, 4))

hasil = 'belum'
for i in xrange(iteration):
	if hasil == 'menang':
		break

	#start player
	player = np.asarray([0,0])
	board = init_board(player)

	for j in xrange(maxstep):
		poslama = np.zeros((2))
		poslama[:] = player
		choice, player = action(qtab, player)
		R = calcreward(poslama, player, target)

		#update q table
		qtab[int(poslama[0]), int(poslama[1]), choice] =  qtab[int(poslama[0]), int(poslama[1]), choice] + lrate*(R + \
		(discount**j)*(np.max(qtab[int(poslama[0]), int(poslama[1])] -  qtab[int(poslama[0]), int(poslama[1]), choice])))

		#update board
		#print 'cek posisi lama', poslama, player
		board[int(poslama[0]), int(poslama[1])] = 0
		board[player[0], player[1]] = playersym
		print board
		#break if landed in trap

		trapped = False
		#check if trapped
		for k in xrange(len(trap)):
			if np.array_equal(player, trap[k]):
				trapped = True
		if trapped:
			break
print np.max(qtab)

#check solution

#start player
player = np.asarray([0,0])
board = init_board(player)
move = []

for j in xrange(maxstep):
	poslama = np.zeros((2))
	poslama[:] = player
	nextmove, player = realaction(qtab, player)
	move.append(nextmove)

	#update board
	board[int(poslama[0]), int(poslama[1])] = 0
	board[player[0], player[1]] = playersym
	print board

	trapped = False
	#check if trapped
	for i in xrange(len(trap)):
		if np.array_equal(player, trap[i]):
			trapped = True
	if trapped:
		print 'FAILED'
		break
	if np.array_equal(player, target):
		break

print 'sukses dalam step ke', j
print 'rincian move nya adalah'
print move
