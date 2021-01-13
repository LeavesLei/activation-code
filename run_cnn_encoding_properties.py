
#-*- coding:utf-8 _*-
""" 
@author:Leaves
@file: run_cnn_encoding_properties.py
@time: 2020/09/09
"""
import argparse
import gc

from activation_code_methods import *
from load_data import *
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import load_model


parser = argparse.ArgumentParser()
parser.add_argument('--architecture', type=str, default='vgg16', help='network architecture')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--mnist_path', type=str, default='mnist.npz', help='MNIST path')
parser.add_argument('--cifar10_path', type=str, default='cifar-10-batches-py', help='CIFAR10 path')
parser.add_argument('--load_path', type=str, default='model/cnn/model_', help='load model path')
parser.add_argument('--save_path', type=str, default='result_new/cnn/result_list_', help='save path')
args = parser.parse_args()

network_architecture = args.architecture
dataset = args.dataset
load_path = args.load_path
save_path = args.save_path

num_classes = 10
n_clusters = num_classes

if dataset == "cifar10":
    (x_train, y_train), (x_test, y_test) = load_cifar10(args.cifar10_path)
    x_train = x_train.reshape(50000, 32, 32, 3)
    x_test = x_test.reshape(10000, 32, 32, 3)
elif dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = load_mnist(path=args.mnist_path, flatten=False)
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)

# generate non-one-hot label for clustering
train_label_scalar = np.argmax(y_train, axis=1).squeeze()
test_label_scalar = np.argmax(y_test, axis=1).squeeze()

input_shape = x_train.shape[1:]

model = load_model(load_path + network_architecture + '_' + dataset + '_adam' + '.h5')

print('architecture: ' + network_architecture)
print('dataset: ' + dataset)

train_score = model.evaluate(x_train, y_train, verbose=0)
print("train loss: ", train_score[0])
print("train accuracy: ", train_score[1])
test_score = model.evaluate(x_test, y_test, verbose=0)
print("test loss: ", test_score[0])
print("test accuracy: ", test_score[1])
save_list([train_score[0], train_score[1], test_score[0], test_score[1]], save_path + network_architecture + '_acc_and_loss_' + dataset)

# Train activation code
train_activation_codes, test_activation_codes = compute_activation_code_for_cnn(x_train, x_test, model=model)
print('neural code dimension: ' + str(train_activation_codes.shape[1]))

# Train redundancy ratio
train_redundancy_ratio = (train_activation_codes.shape[0] - np.unique(train_activation_codes, axis=0).shape[0]) / x_train.shape[0]
print('train redundancy ratio: ' + str(train_redundancy_ratio))

# Test activation code
test_redundancy_ratio = (test_activation_codes.shape[0] - np.unique(test_activation_codes, axis=0).shape[0]) / x_test.shape[0]
print('test redundancy ratio: ' + str(test_redundancy_ratio))

save_list([train_redundancy_ratio, test_redundancy_ratio], save_path + network_architecture + '_redundancy_ratio_' + dataset)
# compute clustering accuracy with kmeans

train_cluster_result = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(train_activation_codes)
train_clustering_accuracy_kmeans = compute_clustering_accuracy(train_cluster_result, train_label_scalar)

test_cluster_result = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(test_activation_codes)
test_clustering_accuracy_kmeans = compute_clustering_accuracy(test_cluster_result, test_label_scalar)

print("train_clustering_accuracy_kmeans: " + str(train_clustering_accuracy_kmeans))
save_list([train_clustering_accuracy_kmeans], save_path + network_architecture + '_train_clustering_accuracy_kmeans_' + dataset)
print("test_clustering_accuracy_kmeans: " + str(test_clustering_accuracy_kmeans))
save_list([test_clustering_accuracy_kmeans], save_path + network_architecture + '_test_clustering_accuracy_kmeans_' + dataset)

# compute clusterisng accuracy with KNN
neigh = KNeighborsClassifier(n_neighbors=9, metric='hamming').fit(train_activation_codes, train_label_scalar)
knn_pred_result = neigh.predict(test_activation_codes)
smstr = np.nonzero(test_label_scalar - knn_pred_result)
knn_accuracy = 1 - np.shape(smstr[0])[0] / test_label_scalar.shape[0]
print("knn_accuracy: " + str(knn_accuracy))
save_list([knn_accuracy], save_path + network_architecture + '_knn_accuracy_' + dataset)
del neigh
gc.collect()

# compute multiclass logisticRegression
logistic_classifier = OneVsOneClassifier(LogisticRegression(random_state=9)).fit(train_activation_codes, train_label_scalar)
logistic_pred_result = logistic_classifier.predict(test_activation_codes)
smstr = np.nonzero(test_label_scalar - logistic_pred_result)
logistic_accuracy = 1 - np.shape(smstr[0])[0] / test_label_scalar.shape[0]
print("logistic_accuracy: " + str(logistic_accuracy))
save_list([logistic_accuracy], save_path + network_architecture + '_logistic_accuracy_' + dataset)
del logistic_classifier
gc.collect()

result_list = [train_score[0], train_score[1], test_score[0], test_score[1], train_redundancy_ratio, test_redundancy_ratio,
                train_clustering_accuracy_kmeans, test_clustering_accuracy_kmeans,
               knn_accuracy, logistic_accuracy]

save_list(result_list, save_path + network_architecture + '_' + dataset)
