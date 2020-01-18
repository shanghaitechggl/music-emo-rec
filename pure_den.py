from densenet import DenseNet
from train_test import train, test
import torch
from dataset import dataset
from torch.utils.data import DataLoader
model = DenseNet(num_classes=18)
train_set=dataset()
test_set=dataset(mode='test')
criterion = nn.BCEWithLogitsLoss()
# model=model.to(device)
model = train(model, train_set, criterion)
test(model)
