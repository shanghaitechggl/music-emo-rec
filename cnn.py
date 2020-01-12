import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import dataset

class CNN(nn.Module):
	"""docstring for CNN"""
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1=nn.Conv2d(1, 16, 2)
		self.conv2=nn.Conv2d(16, 32, 2)
		self.conv3=nn.Conv2d(32, 18, 2)

	def forward(self, x):
		x=self.conv1(x)
		x=self.conv2(x)
		x=self.conv3(x)
		x=x.view(-1, 18)
		return x

batch_size=128
learning_rate=0.0001
num_epoch=20
device=torch.device('cpu')
train_set=dataset()
train_loader=DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_set=dataset(mode='test')
test_loader=DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
criterion = nn.BCEWithLogitsLoss()
def train(model):
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
	for epoch in range(num_epoch):
		for data, label in train_loader:
			data=data-torch.mean(data)
			data=data/torch.var(data)
			out=model(data)
			loss=criterion(out, label)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		print('epoch {}, loss {}'.format(epoch, loss.item()))
	return model

def test(model):
	correct=0
	total=0
	sigmoid=nn.Sigmoid()
	with torch.no_grad():
		for data, label in test_loader:
			data=data-torch.mean(data)
			data=data/torch.var(data)
			out=model(data)
			out=sigmoid(out)
			out=torch.round(out)
			correct+=torch.sum(out==label).data
			total+=label.size(0)*18
	print(100*correct.float()/total)

net=CNN()
net=train(net)
test(net)
