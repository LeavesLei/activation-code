from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
import torchvision.transforms as transforms
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--begin_repeat', type=int, default=1, help=' begin repeat num')
args = parser.parse_args()

begin_repeat = args.begin_repeat

repeat = 1
#begin_repeat = 1
save_path = '/public/data1/users/leishiye/neural_code/results/sample_size/result_list_sample_size_'
load_path = '/public/data1/users/leishiye/neural_code/models/sample_size/model_sample_size_'
depth = 1

dataset = 'svhn'

num_classes = 10
batch_size = 128
n_clusters = num_classes

width_list = [30, 40, 50]
sample_size_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]


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
        for sample_size in sample_size_list:
            print('sample size: ' + str(sample_size))

            trainloader, testloader, _ = load_svhn_sample_size(dataset, batch_size, train=False, sample_size=sample_size)

            net = torch.load(load_path + str(sample_size) + '_width_' + str(num_neuron) + '_' +
                             dataset + '_depth_' + str(depth) + '_iter' + str(iter + 1)).to(device)

            # evaluation
            train_acc = test(net, trainloader, epoch=1)
            print("train accuracy: ", train_acc)
            test_acc = test(net, testloader, epoch=1)
            print("test accuracy: ", test_acc)

            # compute activation code
            train_activation_codes, train_label_scalar = compute_conv_code_list(trainloader, net)
            test_activation_codes, test_label_scalar = compute_conv_code_list(testloader, net)

            # compute redundancy ratio
            test_redundancy_ratio = (test_activation_codes.shape[0] - np.unique(test_activation_codes, axis=0).shape[
                0]) / test_activation_codes.shape[0]
            train_redundancy_ratio = (train_activation_codes.shape[0] - np.unique(train_activation_codes, axis=0).shape[
                0]) / train_activation_codes.shape[0]

            print("train redundancy ratio: " + str(train_redundancy_ratio))
            print("test redundancy ratio: " + str(test_redundancy_ratio))
            
            # compute clustering accuracy with kmeans
            try:
                train_cluster_result = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(train_activation_codes)
                train_clustering_accuracy_kmeans = compute_clustering_accuracy(train_cluster_result, train_label_scalar, n_cluster=n_clusters)
            except:
                train_clustering_accuracy_kmeans = None
            
            try:
                test_cluster_result = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(test_activation_codes)
                test_clustering_accuracy_kmeans = compute_clustering_accuracy(test_cluster_result, test_label_scalar, n_cluster=n_clusters)
            except:
                test_clustering_accuracy_kmeans = None
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
                                test_redundancy_ratio,
                                train_clustering_accuracy_kmeans, test_clustering_accuracy_kmeans, knn_accuracy,
                                logistic_accuracy])
            
        
        # save
        save_list(result_list, save_path + dataset + '_depth_' + str(depth) + '_width_' + str(num_neuron) + '_iter' + str(iter + 1))
        