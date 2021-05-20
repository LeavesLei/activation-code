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

# Basic hyper-parameters
batch_size = 128
num_epochs = 100
repeat = 1
begin_repeat = args.begin_repeat
save_path = '/public/data1/users/leishiye/neural_code/models/layer_width/layer_width_'
depth = 1
dataset = 'svhn'
lr = 0.01

width_list = [10, 15, 20, 23, 27, 30, 33, 37, 40, 43, 47, 50, 53, 57, 60, 65, 70, 75, 80, 90, 100]

# Load SVHN
trainloader, testloader, num_classes = load_svhn(dataset, batch_size)

print('dataset: ' + dataset)
print('depth: ' + str(depth))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

for iter in np.linspace(begin_repeat-1, begin_repeat + repeat-2, repeat).astype('int'):
    print('repeat: ' + str(iter + 1))
    for num_neuron in width_list:
        print('layer width: ' + str(num_neuron))
        net = MLP(n_classes=num_classes, hidden_units=num_neuron).to(device)
        for epoch in range(num_epochs):
            train(net=net, trainloader=trainloader, epoch=epoch, lr=lr, num_epochs=num_epochs)

        torch.save(net, save_path + str(num_neuron) + '_' + dataset + '_depth_' +str(depth)  + '_iter' + str(iter + 1))

