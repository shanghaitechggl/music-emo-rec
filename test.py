rf = open('record-idx.csv','r')
rs=rf.readlines()
test_idxs=rs[-4]
test_idxs=test_idxs.split(',')
test_idxs=list(map(int, test_idxs))
# print((test_idxs))

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
print(len(train_idxs))
