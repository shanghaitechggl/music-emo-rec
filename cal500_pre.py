import os
import csv
import numpy as np


rf = open('record-idx.csv','r')
rs=rf.readlines()
test_idxs=rs[-4]
test_idxs=test_idxs.split(',')
test_idxs=list(map(int, test_idxs))
# data=open('F:/a-workspace/python/datasets/music-emo-rec/music-data-v4.csv','r')
data=open('F:/a-workspace/python/datasets/music-emo-rec/prodAudios_server','r')
lines=data.readlines()

#删除无效数据
length=len(lines)
print(length)
i=0
while i < length:
	while len(lines[i])<10000:
		lines.pop(i)
		length-=1
	i+=1
print(len(lines))
print(length)

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

ls=[16, 32, 64]
for out_len in ls:
	out_file=str(out_len)+'.csv'
	out_file=open(out_file, 'w', encoding='utf8', newline='')
	writer=csv.writer(out_file)
	for i in range(len(idxs)):
		line=lines[idxs[i]]
		line=line.split(' ')
		line=line[0:-1]
		line=list(map(float, line))
		length=len(line)
		num=int(length/out_len)
		out_data=[]
		for j in range(out_len):
			out_data+=[np.mean(line[j*num:(j+1)*num])]
		writer.writerow(out_data)
		print('长度为{}, 第{}个数据'.format(out_len, i))
	out_file.close()


label_file='F:/a-workspace/python/datasets/music-emo-rec/labels-v5.csv'
label_file=open(label_file, 'r')
lines=label_file.readlines()
length=len(lines)
out_file=open('label.csv', 'w', encoding='utf8', newline='')
writer=csv.writer(out_file)
for i in range(length)
