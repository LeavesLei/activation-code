#-*- coding:utf-8 _*-  

# encoding: utf-8
# @File  : run_layer_width_encoding_properties.py
# @Author: LeavesLei
# @Date  : 2020/8/13

from load_data import *
from activation_code_methods import *
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--begin_repeat', type=int, default=1, help=' begin repeat num')
parser.add_argument('--repeat', type=int, default=2, help='repeat times')
parser.add_argument('--mnist_path', type=str, default='mnist.npz', help='MNIST path')
parser.add_argument('--cifar10_path', type=str, default='cifar-10-batches-py', help='CIFAR10 path')
parser.add_argument('--load_path', type=str, default='model/layer_width/model_layer_width_', help='load model path')
parser.add_argument('--save_path', type=str, default='result_new/layer_width/result_list_layer_width_', help='save path')
parser.add_argument('--depth', type=int, default=5, help='depth')
args = parser.parse_args()


repeat = args.repeat
begin_repeat = args.begin_repeat
save_path = args.save_path
load_path = args.load_path
depth = args.depth

dataset = args.dataset

num_classes = 10
n_clusters = num_classes
width_list = [3, 7, 10, 15, 20, 23, 27, 30, 33, 37, 40, 43, 47, 50, 53, 57, 60, 65, 70, 75, 80, 90, 100]

if dataset == "cifar10":
    (x_train, y_train), (x_test, y_test) = load_cifar10(args.cifar10_path)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    width_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
                  550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
# laod MNIST
elif dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = load_mnist(path=args.mnist_path, flatten=True)

num_train = int(x_train.shape[0] * 0.8)
num_val = x_train.shape[0] - num_train
mask = list(range(num_train, num_train+num_val))
x_val = x_train[mask]
y_val = y_train[mask]

mask = list(range(num_train))
x_train = x_train[mask]
y_train = y_train[mask]

train_label_scalar = np.argmax(y_train, axis=1).squeeze()
val_label_scalar = np.argmax(y_val, axis=1).squeeze()
test_label_scalar = np.argmax(y_test, axis=1).squeeze()

input_shape = x_train.shape[1:]

print('dataset: ' + dataset)
print('depth: ' + str(depth))

for iter in np.linspace(begin_repeat-1, begin_repeat + repeat-2, repeat).astype('int'):
    print('repeat: ' + str(iter + 1))
    result_list = []
    for num_neuron in width_list:
        print('layer width: ' + str(num_neuron))
        mlp = load_model(load_path + str(num_neuron) + '_' + dataset + '_depth_' + str(depth) + '_iter' + str(iter + 1) + '.h5')

        # evaluation
        train_score = mlp.evaluate(x_train, y_train, verbose=0)
        print("train loss: ", train_score[0])
        print("train accuracy: ", train_score[1])
        test_score = mlp.evaluate(x_test, y_test, verbose=0)
        print("test loss: ", test_score[0])
        print("test accuracy: ", test_score[1])

        # compute activation code
        train_activation_codes, test_activation_codes = compute_activation_code_for_mlp(x_train, x_test, model=mlp)

        # compute redundancy ratio
        test_redundancy_ratio = (test_activation_codes.shape[0] - np.unique(test_activation_codes, axis=0).shape[0]) / x_test.shape[0]
        train_redundancy_ratio = (train_activation_codes.shape[0] - np.unique(train_activation_codes, axis=0).shape[0]) / x_train.shape[0]

        print("train redundancy ratio: " + str(train_redundancy_ratio))
        print("test redundancy ratio: " + str(test_redundancy_ratio))

        # compute clustering accuracy with kmeans
        train_cluster_result = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(train_activation_codes)
        train_clustering_accuracy_kmeans = compute_clustering_accuracy(train_cluster_result, train_label_scalar)

        test_cluster_result = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(test_activation_codes)
        test_clustering_accuracy_kmeans = compute_clustering_accuracy(test_cluster_result, test_label_scalar)

        print("train_clustering_accuracy_kmeans: " + str(train_clustering_accuracy_kmeans))
        print("test_clustering_accuracy_kmeans: " + str(test_clustering_accuracy_kmeans))

        # compute clusterisng accuracy with KNN
        neigh = KNeighborsClassifier(n_neighbors=9, metric='hamming').fit(train_activation_codes, train_label_scalar)
        knn_pred_result = neigh.predict(test_activation_codes)
        smstr = np.nonzero(test_label_scalar - knn_pred_result)
        knn_accuracy = 1 - np.shape(smstr[0])[0] / test_label_scalar.shape[0]

        print("knn_accuracy: " + str(knn_accuracy))

        # compute multiclass logisticRegression
        logistic_classifier = OneVsOneClassifier(LogisticRegression(random_state=9)).fit(train_activation_codes, train_label_scalar)
        logistic_pred_result = logistic_classifier.predict(test_activation_codes)
        smstr = np.nonzero(test_label_scalar - logistic_pred_result)
        logistic_accuracy = 1 - np.shape(smstr[0])[0] / test_label_scalar.shape[0]

        print("logistic_accuracy: " + str(logistic_accuracy))

        result_list.extend([train_score[0], train_score[1], test_score[0], test_score[1], train_redundancy_ratio, test_redundancy_ratio,
                            train_clustering_accuracy_kmeans, test_clustering_accuracy_kmeans, knn_accuracy, logistic_accuracy])

    # save
    save_list(result_list, save_path + dataset + '_depth_' +str(depth) + '_iter' + str(iter + 1))
