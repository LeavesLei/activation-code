from utils import compute_conv_code_list, test
import sys
sys.path.append("..") 

import torch, os
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from vgg import VGG16
from utils import *

num_classes=200
input_channel = 3
layer_width = 64
batch_size = 128
data_dir = '/public/data1/users/leishiye/datasets'


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

image_datasets = dict()
image_datasets['train'] = TinyImageNet(data_dir, train=True, transform=data_transforms['train'])
image_datasets['test'] = TinyImageNet(data_dir, train=False, transform=data_transforms['test'])
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128, shuffle=True, num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

print("dataset size: ")
print(dataset_sizes)

trainloader = dataloaders['train']
testloader = dataloaders['test']

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
net = VGG16(n_classes=num_classes, input_channel=input_channel, layer_width=layer_width).to(device)

# Hyper-parameters
lr = 0.001
start_epoch = 1
num_epochs = 40
elapsed_time = 0
for epoch in range(start_epoch, start_epoch + num_epochs):

    #Training
    start_time = time.time()
    train(net=net, trainloader=trainloader, epoch=epoch, lr=lr, num_epochs=num_epochs)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

print("Test accuracy: ")
test_acc = test(net, testloader, epoch=1)

# compute activation code
train_activation_codes, _ = compute_conv_code_list(trainloader, net)
test_activation_codes, _ = compute_conv_code_list(testloader, net)

# compute redundancy ratio
test_redundancy_ratio = (test_activation_codes[0].shape[0] - np.unique(test_activation_codes[0], axis=0).shape[0]) / dataset_sizes['test']
train_redundancy_ratio = (train_activation_codes[0].shape[0] - np.unique(train_activation_codes[0], axis=0).shape[0]) / dataset_sizes['train']

print("train redundancy ratio: " + str(train_redundancy_ratio))
print("test redundancy ratio: " + str(test_redundancy_ratio))

# compute redundancy ratio
test_redundancy_ratio = (test_activation_codes[1].shape[0] - np.unique(test_activation_codes[1], axis=0).shape[0]) / dataset_sizes['test']
train_redundancy_ratio = (train_activation_codes[1].shape[0] - np.unique(train_activation_codes[1], axis=0).shape[0]) / dataset_sizes['train']

print("train redundancy ratio: " + str(train_redundancy_ratio))
print("test redundancy ratio: " + str(test_redundancy_ratio))