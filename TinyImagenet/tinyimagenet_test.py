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
data_dir = '/public/data1/users/leishiye/datasets/tiny-imagenet-200/'
num_workers = {'train' : 4,'val'   : 2,'test'  : 2}
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
    ])
}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                  for x in ['train', 'test']}
dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers[x])
                  for x in ['train', 'test']}
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
lr = 0.01
start_epoch = 1
num_epochs = 20
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
train_activation_codes = compute_conv_code_list(trainloader, net)[0]
test_activation_codes = compute_conv_code_list(testloader, net)[0]

# compute redundancy ratio
test_redundancy_ratio = (test_activation_codes.shape[0] - np.unique(test_activation_codes, axis=0).shape[0]) / test_activation_codes.shape[0]
train_redundancy_ratio = (train_activation_codes.shape[0] - np.unique(train_activation_codes, axis=0).shape[0]) / train_activation_codes.shape[0]

print("train redundancy ratio: " + str(train_redundancy_ratio))
print("test redundancy ratio: " + str(test_redundancy_ratio))