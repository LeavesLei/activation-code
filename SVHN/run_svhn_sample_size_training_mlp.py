import torch, os
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from utils import *
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--begin_repeat', type=int, default=1, help=' begin repeat num')
args = parser.parse_args()

begin_repeat = args.begin_repeat


# Basic hyper-parameters
batch_size = 128
num_epochs = 100
repeat = 1
#begin_repeat = 1
save_path = '/public/data1/users/leishiye/neural_code/models/sample_size/model_sample_size_'
depth = 1
dataset = 'svhn'
num_classes = 10
lr = 0.01

width_list = [30, 40, 50]

sample_size_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]

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
            net = MLP(n_classes=num_classes, hidden_units=num_neuron).to(device)

            # training set
            trainloader, _, _ = load_svhn(dataset, batch_size, train=True, sample_size=sample_size)
            for epoch in range(num_epochs):
                train(net=net, trainloader=trainloader, epoch=epoch, lr=lr, num_epochs=num_epochs)

            torch.save(net, save_path + str(sample_size) + '_width_' + str(num_neuron) + '_' + dataset + '_depth_' +
                     str(depth) + '_iter' + str(iter + 1))

        
