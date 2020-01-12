from torch.utils.data import Dataset
import torch
import csv

train_num=2560
test_num=3219-train_num
data_num=16
width=int(data_num**0.5)

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

class dataset(Dataset):
	"""docstring for dataset"""
	def __init__(self, data_file=str(data_num)+'.csv', label_file='label.csv', mode='train'):
		super(dataset, self).__init__()
		self.labels=get_label(label_file)
		data_file=open(data_file, 'r')
		self.music=list(csv.reader(data_file))
		if mode=='train':
			self.len=train_num
			self.start=0
		else:
			self.len=test_num
			self.start=train_num

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		data=self.music[self.start+idx]
		data=list(map(float, data))
		data=torch.Tensor(data).view(1, width, width)
		label=self.labels[self.start+idx]
		return data, label



