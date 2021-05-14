#-*- coding:utf-8 _*-  

# encoding: utf-8
# @File  : run_training_time_encoding_properties.py
# @Author: LeavesLei
# @Date  : 2020/8/13
import sys
sys.path.append("..") 
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
import torchvision.transforms as transforms
from vgg import VGG16
from utils import *

repeat = 5
begin_repeat = 3
input_channel = 3
batch_size = 128
save_path = '/public/data1/users/leishiye/neural_code/results/training_time/result_list_training_process_'
load_path = '/public/data1/users/leishiye/neural_code/models/training_time/model_training_process_'
depth = 1

dataset = 'tinyimagenet'

num_classes = 200
n_clusters = num_classes

width_list = width_list = [64, 128]
output_epoch_list = [0, 1, 2, 3, 6, 8, 10, 13, 17, 20, 25, 30, 35, 40]

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

print('dataset: ' + dataset)
print('depth: ' + str(depth))

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

for iter in np.linspace(begin_repeat-1, begin_repeat + repeat-2, repeat).astype('int'):
    print('repeat: ' + str(iter + 1))
    for num_neuron in width_list:
        print('layer width: ' + str(num_neuron))
        result_list = []

        # training according to training epoch list
        for training_epoch in output_epoch_list:
            net = torch.load(load_path + str(training_epoch) + '_width_' + str(num_neuron) + '_' + dataset + '_depth_' + str(depth) + '_iter' + str(iter + 1)).to(device)

            # evaluation
            train_acc = test(net, trainloader, epoch=1)
            print("train accuracy: ", train_acc)
            test_acc = test(net, testloader, epoch=1)
            print("test accuracy: ", test_acc)
            
            net.train()
            # compute activation code
            train_activation_codes, train_label_scalar = compute_conv_code_list(trainloader, net)
            test_activation_codes, test_label_scalar = compute_conv_code_list(testloader, net)

            train_activation_codes = train_activation_codes[1]
            test_activation_codes = test_activation_codes[1]
            
            # compute redundancy ratio
            test_redundancy_ratio = (test_activation_codes.shape[0] - np.unique(test_activation_codes, axis=0).shape[
                0]) / dataset_sizes['test']
            train_redundancy_ratio = (train_activation_codes.shape[0] - np.unique(train_activation_codes, axis=0).shape[
                0]) / dataset_sizes['train']

            print("train redundancy ratio: " + str(train_redundancy_ratio))
            print("test redundancy ratio: " + str(test_redundancy_ratio))
            
            
            # compute clustering accuracy with kmeans
            train_cluster_result = KMeans(n_clusters=200, random_state=9).fit_predict(train_activation_codes)
            train_clustering_accuracy_kmeans = compute_clustering_accuracy(train_cluster_result, train_label_scalar, n_cluster=n_clusters)

            test_cluster_result = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(test_activation_codes)
            test_clustering_accuracy_kmeans = compute_clustering_accuracy(test_cluster_result, test_label_scalar, n_cluster=n_clusters)

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

            result_list.extend([None, train_acc, None, test_acc, train_redundancy_ratio,
                                test_redundancy_ratio, train_clustering_accuracy_kmeans, test_clustering_accuracy_kmeans,
                                knn_accuracy, logistic_accuracy])
            
        
        # save
        save_list(result_list,
                  save_path + dataset + '_depth_' + str(depth) + '_width_' + str(num_neuron) + '_iter' + str(iter + 1))
        