import os
import csv
import numpy as np

rf = open('record-idx.csv','r')
rs=rf.readlines()
test_idxs=rs[-4]
test_idxs=test_idxs.split(',')
test_idxs=list(map(int, test_idxs))

#构造训练集下标
train_idxs=[]
length=len(test_idxs)
for i in range(3219):
	flag=False
	for j in range(length):
		if i==test_idxs[j]:
			flag=True
			break
	if not flag:
		train_idxs+=[i]

#全部下标
idxs=train_idxs+test_idxs

label_file='F:/a-workspace/python/datasets/music-emo-rec/labels-v5.csv'
label_file=open(label_file, 'r')
lines=label_file.readlines()
length=len(lines)
out_file=open('label.csv', 'w', encoding='utf8', newline='')
writer=csv.writer(out_file)
for i in range(length):
	ll=lines[idxs[i]]
	ll=ll.split(',')
	ll=list(map(int, map(float, ll)))
	writer.writerow(ll)
out_file.close()
label_file.close()
