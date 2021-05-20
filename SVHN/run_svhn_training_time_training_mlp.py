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
repeat = 1
#begin_repeat = 1
num_epochs = 100
save_path = '/public/data1/users/leishiye/neural_code/models/training_time/model_training_process_'
depth = 1
dataset = 'svhn'
lr = 0.01

# Load SVHN
trainloader, testloader, num_classes = load_svhn(dataset, batch_size)

width_list = [30, 40, 50]
output_epoch_list = [1, 2, 3, 6, 10, 15, 30, 40, 60, 70, 90, 100]

training_epoch_list = []
for i in range(len(output_epoch_list)):
    if i == 0:
        training_epoch_list.append(output_epoch_list[i] - 0)
    else:
        training_epoch_list.append(output_epoch_list[i] - output_epoch_list[i-1])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

for iter in np.linspace(begin_repeat-1, begin_repeat + repeat-2, repeat).astype('int'):
    print('repeat: ' + str(iter + 1))
    for num_neuron in width_list:
        print('layer width: ' + str(num_neuron))
        result_list = []

        net = MLP(n_classes=num_classes, hidden_units=num_neuron).to(device)

        torch.save(net, save_path + str(0) + '_width_' + str(num_neuron) + '_' + dataset + '_depth_' + str(depth) + '_iter' + str(iter + 1))

        # training according to training epoch list
        for index, training_epoch in enumerate(training_epoch_list):
            # training networks
            #mlp.fit(x_train, y_train, batch_size=batch_size, epochs=training_epoch, verbose=1)
            for epoch in range(training_epoch):
                if index == 0:
                    train(net=net, trainloader=trainloader, epoch=epoch, lr=lr, num_epochs=num_epochs)
                else:
                    train(net=net, trainloader=trainloader, epoch=output_epoch_list[index-1] + epoch, lr=lr, num_epochs=num_epochs)
            torch.save(net, save_path + str(output_epoch_list[index]) + '_width_' + str(num_neuron) + '_' + dataset + '_depth_' + str(depth) + '_iter' + str(iter + 1))