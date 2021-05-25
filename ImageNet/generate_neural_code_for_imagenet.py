import argparse
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model', default='resnet50', type=str, metavar='N',
                    help='number of data loading workers (default: 4)')

args = parser.parse_args()

#Load data
valdir = '/public/data0/datasets/imagenet2012/ILSVRC2012_img_val'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=256, shuffle=False,
        num_workers=4, pin_memory=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        #self.model = models.resnext50_32x4d(pretrained = True)
        self.model = models.wide_resnet50_2(pretrained = True)
        self.model.eval()
        self.relu = self.model.relu
        layer1 = list(self.model.layer1)[:]
        layer2 = list(self.model.layer2)[:]
        layer3 = list(self.model.layer3)[:]
        layer4 = list(self.model.layer4)[:]
        self.layer1 = nn.ModuleList(layer1).eval()
        self.layer2 = nn.ModuleList(layer2).eval()
        self.layer3 = nn.ModuleList(layer3).eval()
        self.layer4 = nn.ModuleList(layer4).eval()
    def forward(self, x):
        results = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.relu(x)
        results.append(torch.sum(x, (2,3)).detach().cpu().numpy())
        x = self.model.maxpool(x)
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

resnet50 = ResNet50().to(device)
conv_code_list, label_true = compute_conv_code_list(val_loader, resnet50)

save_dir = '/public/data1/users/leishiye/neural_code/results/neural_code_wideresnet50/wideresnet50_code_layer_'
for i in range(len(conv_code_list)):
    np.save(save_dir+str(i), conv_code_list[i])