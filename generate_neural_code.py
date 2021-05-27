import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

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


def TinyImageNet(root='./path', train=True, transform=None):
    if train:
        path = '{}/tiny-imagenet/train.npz'.format(root)
    else:
        path = '{}/tiny-imagenet/test.npz'.format(root)
    data = np.load(path)
    return Dataset(x=data['images'], y=data['labels'], transform=transform)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        #transforms.RandomRotation(20),
        #transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
}

def compute_conv_code_list(data, net):
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data):
            print(batch_idx)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            #inputs, targets = Variable(inputs), Variable(targets)
            aggregate_code_list = net(inputs)
            len_list = len(aggregate_code_list)
            #aggregate_code_list = [(j>0).detach().cpu().numpy() for j in aggregate_code_list]
            targets = targets.detach().cpu().numpy()
            #output = torch.argmax(output, axis=1).detach().cpu().numpy()
            if batch_idx==0:
                conv_code_list = aggregate_code_list
                label_true = targets
            else:
                conv_code_list = [np.concatenate((conv_code_list[i], aggregate_code_list[i]), axis=0) for i in range(len_list)]
                label_true = np.concatenate((label_true, targets))
    return conv_code_list, label_true

batch_size = 128
data_dir = '/public/data1/users/leishiye/datasets'
image_datasets = dict()
image_datasets['train'] = TinyImageNet(data_dir, train=True, transform=data_transforms['train'])
image_datasets['test'] = TinyImageNet(data_dir, train=False, transform=data_transforms['test'])
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_path = '/public/data1/users/leishiye/neural_code/models/pretrained_model_tinyimagenet/'

class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        model = torch.load(load_path+'resnext_tinyimagenet.pt')
        model.eval()
        self.model = model
        self.relu = model.pretrain.relu
        layer1 = list(model.pretrain.layer1)[:]
        layer2 = list(model.pretrain.layer2)[:]
        layer3 = list(model.pretrain.layer3)[:]
        layer4 = list(model.pretrain.layer4)[:]
        self.layer1 = nn.ModuleList(layer1).eval()
        self.layer2 = nn.ModuleList(layer2).eval()
        self.layer3 = nn.ModuleList(layer3).eval()
        self.layer4 = nn.ModuleList(layer4).eval()
    def forward(self, x):
        results = []
        x = self.model.pretrain.conv1(x)
        x = self.model.pretrain.bn1(x)
        x = self.relu(x)
        results.append(torch.sum(x, (2,3)).detach().cpu().numpy())
        x = self.model.pretrain.maxpool(x)
        for layer in self.layer1:
            x_ = layer.conv1(x)
            x_ = layer.bn1(x_)
            x_ = layer.conv2(x_)
            x_ = layer.bn2(x_)
            x_ = layer.conv3(x_)
            x_ = layer.bn3(x_)
            x_ = layer.relu(x_)
            results.append(torch.sum(x_, (2,3)).detach().cpu().numpy())
            x = layer(x)
        for layer in self.layer2:
            x_ = layer.conv1(x)
            x_ = layer.bn1(x_)
            x_ = layer.conv2(x_)
            x_ = layer.bn2(x_)
            x_ = layer.conv3(x_)
            x_ = layer.bn3(x_)
            x_ = layer.relu(x_)
            results.append(torch.sum(x_, (2,3)).detach().cpu().numpy())
            x = layer(x)
        for layer in self.layer3:
            x_ = layer.conv1(x)
            x_ = layer.bn1(x_)
            x_ = layer.conv2(x_)
            x_ = layer.bn2(x_)
            x_ = layer.conv3(x_)
            x_ = layer.bn3(x_)
            x_ = layer.relu(x_)
            results.append(torch.sum(x_, (2,3)).detach().cpu().numpy())
            x = layer(x)
        for layer in self.layer4:
            x_ = layer.conv1(x)
            x_ = layer.bn1(x_)
            x_ = layer.conv2(x_)
            x_ = layer.bn2(x_)
            x_ = layer.conv3(x_)
            x_ = layer.bn3(x_)
            x_ = layer.relu(x_)
            results.append(torch.sum(x_, (2,3)).detach().cpu().numpy())
            x = layer(x)
        return results

resnet50 = ResNet50().to(device)

conv_code_list, label_true = compute_conv_code_list(dataloaders['test'], resnet50)
neural_code_dir = '/public/data1/users/leishiye/neural_code/results/neural_code_tinyimagenet/neural_code_resnext50/resnext50_test_code_layer_'
for i in range(17):
    layer_code = np.save(neural_code_dir + str(i), conv_code_list[i])
np.save('/public/data1/users/leishiye/neural_code/results/neural_code_tinyimagenet/test_label', label_true)

conv_code_list, label_true = compute_conv_code_list(dataloaders['train'], resnet50)
neural_code_dir = '/public/data1/users/leishiye/neural_code/results/neural_code_tinyimagenet/neural_code_resnext50/resnext50_train_code_layer_'
for i in range(17):
    layer_code = np.save(neural_code_dir + str(i), conv_code_list[i])
np.save('/public/data1/users/leishiye/neural_code/results/neural_code_tinyimagenet/train_label', label_true)
