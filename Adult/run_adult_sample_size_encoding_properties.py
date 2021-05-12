import sys
sys.path.append("..") 
import numpy as np
from load_data import *
from activation_code_methods import *
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture as GMM
from keras.models import load_model
from adult import Adult
import numpy as np
from keras.utils import to_categorical

repeat = 3
begin_repeat = 3
save_path = '/public/data1/users/leishiye/neural_code/results/sample_size/result_list_sample_size_'
load_path = '/public/data1/users/leishiye/neural_code/models/sample_size/model_sample_size_'
depth = 1

dataset = 'adult'

num_classes = 2

n_clusters = num_classes

width_list = [60, 80, 100]
sample_size_list = [100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000]
# Load data
##########################################
trainset = Adult(root='/public/data1/users/leishiye/neural_code/datasets', train=True)
x_train = trainset.x / trainset.x.max(axis=0)
y_train = to_categorical(trainset.y)

num_train = int(x_train.shape[0] * 0.7)
num_test = x_train.shape[0] - num_train
mask = list(range(num_train, num_train+num_test))
x_test = x_train[mask]
y_test = y_train[mask]

mask = list(range(num_train))
x_train = x_train[mask]
y_train = y_train[mask]

test_label_scalar = np.argmax(y_test, axis=1).squeeze()

input_shape = x_train.shape[1:]

print('dataset: ' + dataset)
print('depth: ' + str(depth))

for iter in np.linspace(begin_repeat-1, begin_repeat + repeat-2, repeat).astype('int'):
    print('repeat: ' + str(iter + 1))
    for num_neuron in width_list:
        print('layer width: ' + str(num_neuron))
        result_list = []
        for sample_size in sample_size_list:
            print('sample size: ' + str(sample_size))

            # training set
            x_sub_train = x_train[:sample_size]
            y_sub_train = y_train[:sample_size]
            train_label_scalar = np.argmax(y_sub_train, axis=1).squeeze()

            mlp = load_model(load_path + str(sample_size) + '_width_' + str(num_neuron) + '_' +
                             dataset + '_depth_' + str(depth) + '_iter' + str(iter + 1) + '.h5')

            # evaluation
            train_score = mlp.evaluate(x_sub_train, y_sub_train, verbose=0)
            print("train loss: ", train_score[0])
            print("train accuracy: ", train_score[1])
            test_score = mlp.evaluate(x_test, y_test, verbose=0)
            print("test loss: ", test_score[0])
            print("test accuracy: ", test_score[1])

            # compute activation code
            train_activation_codes, test_activation_codes = compute_activation_code_for_mlp(x_sub_train, x_test, model=mlp)

            # compute redundancy ratio
            test_redundancy_ratio = (test_activation_codes.shape[0] - np.unique(test_activation_codes, axis=0).shape[
                0]) / x_test.shape[0]
            train_redundancy_ratio = (train_activation_codes.shape[0] - np.unique(train_activation_codes, axis=0).shape[
                0]) / x_sub_train.shape[0]

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

            result_list.extend([train_score[0], train_score[1], test_score[0], test_score[1], train_redundancy_ratio,
                                test_redundancy_ratio,
                                train_clustering_accuracy_kmeans, test_clustering_accuracy_kmeans, knn_accuracy,
                                logistic_accuracy])
        # save
        save_list(result_list, save_path + dataset + '_depth_' + str(depth) + '_width_' + str(num_neuron) + '_iter' + str(iter + 1))