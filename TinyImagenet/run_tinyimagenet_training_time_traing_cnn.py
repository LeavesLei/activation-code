import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from vgg import VGG16
from utils import *
import numpy as np


# Basic hyper-parameters
batch_size = 128
repeat = 5
begin_repeat = 1
save_path = '/public/data1/users/leishiye/neural_code/models/training_time/model_training_process_'
depth = 1
dataset = 'tinyimagenet'
input_channel = 3
num_classes = 200
lr = 0.001

width_list = [64, 128]
output_epoch_list = [1, 2, 3, 6, 8, 10, 13, 17, 20, 25, 30, 35, 40]

training_epoch_list = []
for i in range(len(output_epoch_list)):
    if i == 0:
        training_epoch_list.append(output_epoch_list[i] - 0)
    else:
        training_epoch_list.append(output_epoch_list[i] - output_epoch_list[i-1])

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
image_datasets['train'] = TinyImageNet(data_dir, train=True, transform=data_transforms['train'])
image_datasets['test'] = TinyImageNet(data_dir, train=False, transform=data_transforms['test'])
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

print("dataset size: ")
print(dataset_sizes)

trainloader = dataloaders['train']
testloader = dataloaders['test']

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

for iter in np.linspace(begin_repeat-1, begin_repeat + repeat-2, repeat).astype('int'):
    print('repeat: ' + str(iter + 1))
    for num_neuron in width_list:
        print('layer width: ' + str(num_neuron))
        result_list = []

        net = VGG16(n_classes=num_classes, input_channel=input_channel, layer_width=num_neuron).to(device)

        torch.save(net, save_path + str(0) + '_width_' + str(num_neuron) + '_' + dataset + '_depth_' + str(depth) + '_iter' + str(iter + 1))

        # training according to training epoch list
        for index, training_epoch in enumerate(training_epoch_list):

            # training networks
            #mlp.fit(x_train, y_train, batch_size=batch_size, epochs=training_epoch, verbose=1)
            for epoch in range(training_epoch):
                train(net=net, trainloader=trainloader, epoch=epoch, lr=lr, num_epochs=40)
            
            torch.save(net, save_path + str(output_epoch_list[index]) + '_width_' + str(num_neuron) + '_' + dataset + '_depth_' + str(depth) + '_iter' + str(iter + 1))