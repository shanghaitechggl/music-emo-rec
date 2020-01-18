import torch
import torch.nn as nn
batch_size=128
learning_rate=0.0001
num_epoch=20
device = torch.device('cuda:0')
# device=torch.device('cpu')
def train(model, train_set, criterion):
	model=model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
	train_loader=DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
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
	model=model.to(device)
	correct=0
	total=0
	c_pre=0
	t_pre=0
	sigmoid=nn.Sigmoid()
	test_loader=DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
	with torch.no_grad():
		for data, label in test_loader:
			data=data-torch.mean(data)
			data=data/torch.var(data)
			out=model(data)
			out=sigmoid(out)
			out=torch.round(out)
			correct+=torch.sum(out==label).data
			total+=label.size(0)*18
			m, n=out.size()
			for i in range(m):
				for j in range(n):
					if out[i][j]==1:
						t_pre+=1
						if label[i][j]==1:
							c_pre+=1
	print(100*correct.float()/total)
	print(100*float(c_pre)/t_pre)


