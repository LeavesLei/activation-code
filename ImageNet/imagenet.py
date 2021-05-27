  
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


import argparse
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--gpu', default=True, type=int,
                    help='GPU id to use.')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if use_cuda:
                images = images.cuda()
                target = target.cuda()
            # compute output
            output = model(images)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg


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
model = models.vgg19(pretrained=True).to(device)
criterion = nn.CrossEntropyLoss().cuda(args.gpu)

validate(val_loader, model, criterion, args)


######################TEST#######################################

for i, (images, target) in enumerate(val_loader):
    if use_cuda:
        images = images.cuda()
        target = target.cuda()
    break

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        model = models.vgg19(pretrained = True)
        features = list(model.features)[:]
        classifier = list(model.classifier)[:]
        self.features = nn.ModuleList(features).eval()
        self.classifier = nn.ModuleList(classifier).eval() 
        
    def forward(self, x):
        results = []
        for ii,layer in enumerate(self.features):
            x = layer(x)
            if ii in {1,3, 6,8,11,13,15,17,20,22,24,26,29,31,33,35}: 
                #results.append(x.detach().cpu().numpy())
                results.append(torch.sum(x, (2,3)).detach().cpu().numpy())
        x = model.avgpool(x)
        for ii,layer in enumerate(self.classifier):
            x = torch.flatten(x, start_dim=1)
            x = layer(x)
            if ii in {1,4}: 
                results.append(x.detach().cpu().numpy())
        return results

vgg19 = VGG19().to(device)
a = vgg19(images)

class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        model = models.resnet50(pretrained = True)
        model.eval()
        self.relu = model.relu
        layer1 = list(model.layer1)[:]
        layer2 = list(model.layer2)[:]
        layer3 = list(model.layer3)[:]
        layer4 = list(model.layer4)[:]
        self.layer1 = nn.ModuleList(layer1).eval()
        self.layer2 = nn.ModuleList(layer2).eval()
        self.layer3 = nn.ModuleList(layer3).eval()
        self.layer4 = nn.ModuleList(layer4).eval()
    def forward(self, x):
        results = []
        x = model.conv1(x)
        x = model.bn1(x)
        x = self.relu(x)
        results.append(torch.sum(x, (2,3)).detach().cpu().numpy())
        x = model.maxpool(x)
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

conv_code_list, label_true = compute_conv_code_list(val_loader, resnet50)    
             



################################################
"""
model.eval()

with torch.no_grad():
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        if use_cuda:
            images = images.cuda()
            target = target.cuda()
"""
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

conv_code_list, label_true = compute_conv_code_list(val_loader, vgg19)


def compute_conv_code_list(data):
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data):
            print(batch_idx)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            #inputs, targets = Variable(inputs), Variable(targets)
            #aggregate_code_list = net(inputs)
            #len_list = len(aggregate_code_list)
            #aggregate_code_list = [(j>0).detach().cpu().numpy() for j in aggregate_code_list]
            targets = targets.detach().cpu().numpy()
            #output = torch.argmax(output, axis=1).detach().cpu().numpy()
            if batch_idx==0:
                #conv_code_list = aggregate_code_list
                label_true = targets
            else:
                #conv_code_list = [np.concatenate((conv_code_list[i], aggregate_code_list[i]), axis=0) for i in range(len_list)]
                label_true = np.concatenate((label_true, targets))
    return label_true

###################################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression

# compute clustering accuracy with kmeans
import numpy as np

# Load neural code
neural_code_dir = '/public/data1/users/leishiye/neural_code/results/neural_code_vgg19/vgg19_code_layer_'

for i in range(18):
    layer_code = np.load(neural_code_dir + str(i) + '.npy')
    # normalize
    layer_code = layer_code / (np.max(layer_code) - np.min(layer_code))
    if i == 0:
        neural_code = layer_code
    else:
        neural_code = np.hstack((neural_code, layer_code))


n_clusters = 1000
train_cluster_result = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(train_activation_codes)
train_clustering_accuracy_kmeans = compute_clustering_accuracy(train_cluster_result, train_label_scalar)

test_cluster_result = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(test_activation_codes)
test_clustering_accuracy_kmeans = compute_clustering_accuracy(test_cluster_result, test_label_scalar)

print("train_clustering_accuracy_kmeans: " + str(train_clustering_accuracy_kmeans))
print("test_clustering_accuracy_kmeans: " + str(test_clustering_accuracy_kmeans))

# compute clusterisng accuracy with KNN
neigh = KNeighborsClassifier(n_neighbors=9, metric='hamming').fit(train_activation_codes,
                                                                    train_label_scalar)
knn_pred_result = neigh.predict(test_activation_codes)
smstr = np.nonzero(test_label_scalar - knn_pred_result)
knn_accuracy = 1 - np.shape(smstr[0])[0] / test_label_scalar.shape[0]

print("knn_accuracy: " + str(knn_accuracy))

# compute multiclass logisticRegression
logistic_classifier = OneVsOneClassifier(LogisticRegression(solver='liblinear', random_state=9)).fit(train_activation_codes,
                                                                                    train_label_scalar)
logistic_pred_result = logistic_classifier.predict(test_activation_codes)
smstr = np.nonzero(test_label_scalar - logistic_pred_result)
logistic_accuracy = 1 - np.shape(smstr[0])[0] / test_label_scalar.shape[0]

print("logistic_accuracy: " + str(logistic_accuracy))

######################### TinyImageNet #######################################################
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

class VGG(nn.Module):
    def __init__(self, class_num=1000, hidden=4096):
        super(VGG, self).__init__()
        self.pretrain = torchvision.models.vgg19_bn(pretrained=True)
        # for layer in [*self.pretrain.features.children()][:40]:
        #     layer.requires_grad_(False)
        # self.pretrain.features.requires_grad_(False)
        self.pretrain.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Dropout(),
            nn.Linear(hidden, class_num),
        )
    def forward(self, x):
        x = feature = self.pretrain(x)
        x = self.out(x)
        return x, feature

batch_size = 128
data_dir = '/public/data1/users/leishiye/datasets'
image_datasets = dict()
image_datasets['train'] = TinyImageNet(data_dir, train=True, transform=data_transforms['train'])
image_datasets['test'] = TinyImageNet(data_dir, train=False, transform=data_transforms['test'])
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

for i, (images, target) in enumerate(dataloaders['train']):
    if use_cuda:
        images = images.cuda()
        target = target.cuda()
    break

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_path = '/public/data1/users/leishiye/neural_code/models/pretrained_model_tinyimagenet/'

# For VGG19 BN on tinyimagenet
class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        #model = models.vgg19(pretrained = True)
        model = torch.load(load_path+'vgg_tinyimagenet.pt')
        features = list(model.pretrain.features)[:]
        classifier = list(model.pretrain.classifier)[:]
        self.features = nn.ModuleList(features).eval()
        self.classifier = nn.ModuleList(classifier).eval() 
        
    def forward(self, x):
        results = []
        for ii,layer in enumerate(self.features):
            x = layer(x)
            #if ii in {1,3, 6,8,11,13,15,17,20,22,24,26,29,31,33,35}:
            if ii in {2,5,9,12,16,19,22,25,29,32,35,38,42,45,48,51}:
                #results.append(x.detach().cpu().numpy())
                results.append(torch.sum(x, (2,3)).detach().cpu().numpy())
        x = model.pretrain.avgpool(x)
        for ii,layer in enumerate(self.classifier):
            x = torch.flatten(x, start_dim=1)
            x = layer(x)
            if ii in {1,4}: 
                results.append(x.detach().cpu().numpy())
        return results

class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        model = torch.load(load_path+'resnet_tinyimagenet.pt')
        model.eval()
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
        x = model.pretrain.conv1(x)
        x = model.pretrain.bn1(x)
        x = self.relu(x)
        results.append(torch.sum(x, (2,3)).detach().cpu().numpy())
        x = model.pretrain.maxpool(x)
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


class ResNet(nn.Module):
    def __init__(self, class_num=1000):
        super(ResNet, self).__init__()
        self.pretrain = torchvision.models.resnet50(pretrained=True)
        # for layer in [*self.pretrain.children()][:7]:
        #     layer.requires_grad_(False)
        self.pretrain.avgpool.register_forward_hook(self.hook)
        self.feature = 0
        self.pretrain.fc = nn.Linear(2048, class_num)
    def hook(self, module, fea_in, fea_out):
        self.feature = torch.flatten(fea_out)
        return None
    def forward(self, x):
        x = self.pretrain(x)
        return x, self.feature

vgg = VGG19().to(device)
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
conv_code_list, label_true = compute_conv_code_list(dataloaders['test'], resnet50)# vgg)  

#Save neural code
#neural_code_dir = '/public/data1/users/leishiye/neural_code/results/neural_code_tinyimagenet/neural_code_vgg19/vgg19_test_code_layer_'
neural_code_dir = '/public/data1/users/leishiye/neural_code/results/neural_code_tinyimagenet/neural_code_vgg19/vgg19_train_code_layer_'

for i in range(18):
    layer_code = np.save(neural_code_dir + str(i), conv_code_list[i])
np.save('/public/data1/users/leishiye/neural_code/results/neural_code_tinyimagenet/train_label', label_true)