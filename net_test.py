# %load alldata-cnn.py
from torch.utils.data import Dataset
import csv
import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import statistics
import numpy as np
import random
sample_num = 100
total = 3219
start = 0
train_total=2575
data_len=16
width=4
basic_dir = '../data/'
data_dir = basic_dir + 'cal500_ser/cal500/'
label_file = basic_dir + 'labels-v5.csv'
data_file = basic_dir + 'music-data-v8.csv'

batch_size=32
# device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
device=torch.device('cpu')
learning_rate=0.001
num_epochs=1


def get_label(label_file):
    label_file = open(label_file, 'r')
    label_reader = csv.reader(label_file)
    reader_list = list(label_reader)
    create_labels = True
    for line in reader_list:
        if not line:
            continue
        if create_labels:
            labels = [list(map(int, list(map(float, line))))]
            create_labels = False
        else:
            labels.append(list(map(int, list(map(float, line)))))
    return torch.Tensor(labels)

# 用于训练和测试数据在同一个文件夹内，但每一行未处理的raw_data都属于单独一个文件
class MusicDataFour(Dataset):
    def __init__(self, data_dir=data_dir, label_file=label_file,
     transform=None, pic_len=256, start=start, total=train_total, mode='normal'):
        super(MusicDataFour, self).__init__()

        if mode == 'one-hot':
            self.labels = one_hot_label(label_file)
        else:
            self.labels = get_label(label_file)


        self.file_names = os.listdir(path=data_dir)
        self.len = total - start
        self.sample_len = pic_len * pic_len
        self.pic_len = pic_len
        self.transform = transform
        self.start = start
        self.data_dir = data_dir

    def __len__(self):
        # return self.len * sample_num
        return self.len

    def __getitem__(self, idx):
#         print(idx)
        rows = list(csv.reader(open(self.data_dir+self.file_names[idx], 'r')));
        n_row=len(rows)
        row=[];
        for i in range(n_row):
            if len(rows[i])>self.sample_len:
                row=row+rows[i][0:self.sample_len]
            else:
                row=row+rows[i]
#         print(len(row))
        row=list(map(float,row))
        num=int(len(row)/data_len)
        data=[]
        for i in range(data_len):
            data=data+[np.mean(row[i*num:(i+1)*num])]
#         print(data)
        data=data-np.mean(data);
        
        data = torch.Tensor(data).reshape(1,width,width)
        label = torch.Tensor(self.labels[idx])
        return data, label


# 用于训练和测试数据在同一个文件内
class MusicDataThree(Dataset):
    def __init__(self, data_file=data_file, label_file=label_file,
     transform=None, pic_len=256, start=start, total=train_total, mode='normal'):
        super(MusicDataThree, self).__init__()

        if mode == 'one-hot':
            self.labels = one_hot_label(label_file)
        else:
            self.labels = get_label(label_file)

        data_file = open(data_file, 'r')
        self.rows = data_file.readlines()
        self.len = total - start
        self.sample_len = pic_len * pic_len
        self.pic_len = pic_len
        self.transform = transform
        self.start = start

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        row = self.rows[idx].split(',')
        start = 0
        data=list(map(float,row))
        data=torch.Tensor(data).reshape(1,width,width)
        data=data/torch.max(data)
        label = self.labels[idx]
        label=torch.Tensor(label)
        return data, label


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 2),
            nn.Conv2d(32,32,2),
            nn.Conv2d(32,18,2) 
        )

    def forward(self, x):
        x = self.cnn(x)
        x=x.view(-1,18)
#         sigmoid = nn.Sigmoid()
#         x=sigmoid(x)
        return x

class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.fnn = nn.Sequential(
            nn.Linear(16, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 18),
        )

    def forward(self, x):
        x=x.view(-1,16)
        x = self.fnn(x)
        x=x.view(-1,18)
#         sigmoid = nn.Sigmoid()
#         x=sigmoid(x)
        return x

dataset = MusicDataThree()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
test_set = MusicDataThree(start=train_total, total=total)
model = CNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-7)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10,
            threshold=1e-6, factor=0.0001, min_lr=1e-6)

for epoch in range(num_epochs):
    for img, label in dataloader:
        # img,_ = data
        img = (img).to(device)
        label=label.to(device)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, label)
        # print(loss)
        # ===================backward====================
        optimizer.zero_grad()
        scheduler.step(loss)
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.item()))
    torch.save(model.state_dict(), './CNN.pth')

def test(net, dataset=test_set):
    correct = 0
    total = 0
    loss = 0
    
    tp=0
    fp=0
    sigmoid = nn.Sigmoid()
    # threshold = 0
    with torch.no_grad():
        test_loader = DataLoader(dataset=dataset,
                                 batch_size=batch_size, shuffle=False)
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            outputs = sigmoid(outputs)
            outputs = torch.round(outputs)
            total += labels.size(0)*18
            correct += torch.sum(outputs==labels).data
#             tp+=torch.sum((labels==1)==(outputs==1))
#             fp+=torch.sum((labels==0)==(outputs==1))
            height, width=labels.size()
            for i in range(height):
                for j in range(width):
                    if outputs[i][j]==1:
                        if labels[i][j]==1:
                            tp+=1
                        else:
                            fp+=1

    print('Accuracy of the network on the test images: {} %'.format(
            100*correct.float() / total))
    print('Precision:{}'.format(100*float(tp)/(tp+fp)))
test(model, dataset=test_set)