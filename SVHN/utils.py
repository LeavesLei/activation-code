import sys
import time
import torch
import torch.optim as optim
import math
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import torchvision
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")

criterion = nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()

def compute_conv_code_list(data, net):
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            *aggregate_code_list, output = net(inputs)
            aggregate_code_list = [(j>0).detach().cpu().numpy() for j in aggregate_code_list]
            targets = targets.detach().cpu().numpy()
            output = torch.argmax(output, axis=1).detach().cpu().numpy()
            if batch_idx==0:
                conv_code_list = aggregate_code_list
                label_true = targets
            else:
                conv_code_list = [np.concatenate((conv_code_list[i], aggregate_code_list[i]), axis=0) for i in range(1)]
                label_true = np.concatenate((label_true, targets))
    return conv_code_list[0], label_true

def compute_diffusion(data, net, num_classes=10):
    conv_code, label_true = compute_conv_code(data, net)

    inner_class_distance = []
    class_pair_num = []
    sum_inter_class_distance = 0
    pairwise_num = conv_code.shape[0] * (conv_code.shape[0] - 1)
    # average distance
    distA=pdist(conv_code, metric='euclidean')
    distB = squareform(distA)
    distB = distB[~np.eye(distB.shape[0],dtype=bool)].reshape(distB.shape[0],-1)
    std_distance = np.std(distB)
    distB = np.square(distB)
    average_distance = np.mean(1./distB)

    for i in range(num_classes):
        distA=pdist(conv_code[[label_true==i]], metric='euclidean')
        distB = squareform(distA)
        distB = distB[~np.eye(distB.shape[0],dtype=bool)].reshape(distB.shape[0],-1)
        distB = np.square(distB)
        inner_class_distance.append(np.mean(1./distB))
        class_pair_num.append(conv_code[[label_true==i]].shape[0] * (conv_code[[label_true==i]].shape[0] - 1))

        sum_inter_class_distance = sum_inter_class_distance + class_pair_num[i] * inner_class_distance[i]
        pairwise_num = pairwise_num - class_pair_num[i]

    inner_distance = sum_inter_class_distance / np.sum(class_pair_num)
    inter_distance = (average_distance * conv_code.shape[0] * (conv_code.shape[0] - 1) - sum_inter_class_distance) / pairwise_num
    return average_distance, inner_distance, inter_distance, std_distance

def train(net, trainloader, epoch=1, lr=0.1, num_epochs=100):
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0

    optimizer = optim.SGD(net.parameters(), lr=learning_rate(lr, epoch), momentum=0.9, weight_decay=5e-4)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, learning_rate(lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda().reshape(-1, 32*32*3), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        *_, outputs = net(inputs)
        loss = criterion(outputs, targets) # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    len(trainloader), loss.item(), 100.*correct/total))
        sys.stdout.flush()

def test(net, testloader, epoch=1):
    global best_acc
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda().reshape(-1, 32*32*3), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            *_,  outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.*correct/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))
    return acc.detach().cpu().numpy()


def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 90):
        optim_factor = 3
    elif(epoch > 60):
        optim_factor = 2
    elif(epoch > 40):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s

def save_list(file_name, path):
    file = open(path, 'w')
    for fp in file_name:
        file.write(str(fp))
        file.write('\n')
    file.close()


def load_list(path):
    data = []
    file_handler =open(path, mode='r')
    contents = file_handler.readlines()
    for name in contents:
        name = name.strip('\n')
        data.append(float(name))
    return data


def compute_clustering_accuracy(cluster_result, label, n_cluster=200):
    partition_matrix = contingency_matrix(label, cluster_result)
    _, label_mapping = linear_sum_assignment(-partition_matrix)
    for i in range(n_cluster):
        cluster_result[cluster_result == label_mapping[i]] = n_cluster + i
    cluster_result = cluster_result - n_cluster
    smstr = np.nonzero(label - cluster_result)
    clustering_accuracy = 1 - np.shape(smstr[0])[0] / label.shape[0]
    return clustering_accuracy


mean = {
    'svhn': (0.5, 0.5, 0.5),
}

std = {
    'svhn': (1., 1., 1.),
}

def load_svhn(dataset='svhn', batch_size=128):

    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[dataset], std[dataset]),
    ]) # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean[dataset], std[dataset]),
    ])

    print("| Preparing SVHN dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.SVHN(root='/public/data1/users/leishiye/datasets/svhn', split='train', download=True, transform=transform_train)
    testset = torchvision.datasets.SVHN(root='/public/data1/users/leishiye/datasets/svhn', split='test', download=False, transform=transform_test)
    num_classes = 10

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes


def load_svhn_sample_size(dataset='svhn', batch_size=128, train=False, sample_size=10000):

    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[dataset], std[dataset]),
    ]) # meanstd transformation

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean[dataset], std[dataset]),
    ])

    print("| Preparing SVHN dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.SVHN(root='/public/data1/users/leishiye/datasets/svhn', split='train', download=True, transform=transform_train)
    testset = torchvision.datasets.SVHN(root='/public/data1/users/leishiye/datasets/svhn', split='test', download=False, transform=transform_test)
    trainset.data = trainset.data[:sample_size]
    trainset.labels = trainset.labels[:sample_size]
    if train:
        # extend the dataset
        expansion_factor = 50000 // sample_size
        trainset.data = np.tile(trainset.data, (expansion_factor, 1, 1, 1))
        trainset.labels = np.tile(trainset.labels, (expansion_factor))

    num_classes = 10

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    return trainloader, testloader, num_classes


class MLP(nn.Module):
    def __init__(self, n_classes=10, hidden_units=100):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(32*32*3, hidden_units)
        self.layer2 = nn.Linear(hidden_units, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        features_1 = out
        out = self.layer2(out)

        return features_1, out