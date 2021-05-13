# encoding: utf-8
# @File  : run_sample_size_train_mlp.py
# @Author: LeavesLei
# @Date  : 2020/8/13

import torch, os
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from vgg import VGG16
from utils import *
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--begin_repeat', type=int, default=1, help=' begin repeat num')
args = parser.parse_args()

begin_repeat = args.begin_repeat

class Dataset():
    def __init__(self, x, y, transform=None):
        assert(len(x) == len(y))
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.transform is not None:
            x = self.transform( Image.fromarray(x) )
            # x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.x)


def TinyImageNet(root='./path', train=True, transform=None, sample_size=100000):
    if train:
        path = '{}/tiny-imagenet/train.npz'.format(root)
    else:
        path = '{}/tiny-imagenet/test.npz'.format(root)

    data = np.load(path)
    if train:
        # extend the dataset
        expansion_factor = 100000 // sample_size
        x_sub_train = data['images'][range(0,100000,100000//sample_size)]
        y_sub_train = data['labels'][range(0,100000,100000//sample_size)]
        x_sub_train_expansion = np.tile(x_sub_train, (expansion_factor, 1))
        y_sub_train_expansion = np.tile(y_sub_train, (expansion_factor))
        return Dataset(x=x_sub_train_expansion, y=y_sub_train_expansion, transform=transform)
    else:
        return Dataset(x=data['images'], y=data['labels'], transform=transform)

# Basic hyper-parameters
batch_size = 128
num_epochs = 40
repeat = 1
#begin_repeat = 1
save_path = '/public/data1/users/leishiye/neural_code/models/sample_size/model_sample_size_'
depth = 1
dataset = 'tinyimagenet'
input_channel = 3
num_classes = 200
lr = 0.001

width_list = [64, 128]

sample_size_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
# data loading
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])
}

data_dir = '/public/data1/users/leishiye/datasets'
image_datasets = dict()
"""
image_datasets['train'] = TinyImageNet(data_dir, train=True, transform=data_transforms['train'])
image_datasets['test'] = TinyImageNet(data_dir, train=False, transform=data_transforms['test'])
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128, shuffle=True, num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

print("dataset size: ")
print(dataset_sizes)
"""

trainloader = dataloaders['train']
testloader = dataloaders['test']

print('dataset: ' + dataset)
print('depth: ' + str(depth))

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

for iter in np.linspace(begin_repeat-1, begin_repeat + repeat-2, repeat).astype('int'):
    print('repeat: ' + str(iter + 1))
    for num_neuron in width_list:
        print('layer width: ' + str(num_neuron))
        for sample_size in sample_size_list:
            print('sample size: ' + str(sample_size))

            # building model
            net = VGG16(n_classes=num_classes, input_channel=input_channel, layer_width=num_neuron).to(device)

            # training set
            image_datasets['train'] = TinyImageNet(data_dir, train=True, transform=data_transforms['train'])
            trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=128, shuffle=True, num_workers=4)
            for epoch in range(num_epochs):
                train(net=net, trainloader=trainloader, epoch=epoch, lr=lr, num_epochs=num_epochs)

            torch.save(net, save_path + str(sample_size) + '_width_' + str(num_neuron) + '_' + dataset + '_depth_' +
                     str(depth) + '_iter' + str(iter + 1))

        
