#-*- coding:utf-8 _*-  

# encoding: utf-8
# @File  : run_training_time_encoding_properties.py
# @Author: LeavesLei
# @Date  : 2020/8/13

from load_data import *
from activation_code_methods import *
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from keras.models import load_model
from keras.datasets import imdb
import numpy as np
from keras.utils import to_categorical

repeat = 5
begin_repeat = 1
save_path = '/public/data1/users/leishiye/neural_code/results/training_time/result_list_training_process_'
load_path = '/public/data1/users/leishiye/neural_code/models/training_time/model_training_process_'
depth = 1

dataset = 'imdb'

num_classes = 2
n_clusters = num_classes

width_list = [60, 80] #[100]
output_epoch_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20]

# Load data
##########################################
# number of most-frequent words 
nb_words = 10000

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=nb_words)
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])
def vectorize_sequences(sequences, dimension=nb_words):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Convert training data to bag-of-words:
x_train = vectorize_sequences(x_train)
x_test = vectorize_sequences(x_test)

# Convert labels from integers to floats:
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
###########################################

train_label_scalar = np.argmax(y_train, axis=1).squeeze()
test_label_scalar = np.argmax(y_test, axis=1).squeeze()

input_shape = x_train.shape[1:]

print('dataset: ' + dataset)
print('depth: ' + str(depth))

for iter in np.linspace(begin_repeat-1, begin_repeat + repeat-2, repeat).astype('int'):
    print('repeat: ' + str(iter + 1))
    for num_neuron in width_list:
        print('layer width: ' + str(num_neuron))
        result_list = []

        # training according to training epoch list
        for training_epoch in output_epoch_list:
            # load model
            mlp = load_model(load_path + str(training_epoch) + '_width_' + str(num_neuron) + '_' + dataset + '_depth_' + str(depth) + '_iter' + str(iter + 1) + '.h5')

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
            test_redundancy_ratio = (test_activation_codes.shape[0] - np.unique(test_activation_codes, axis=0).shape[
                0]) / x_test.shape[0]
            train_redundancy_ratio = (train_activation_codes.shape[0] - np.unique(train_activation_codes, axis=0).shape[
                0]) / x_train.shape[0]

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
                                test_redundancy_ratio, train_clustering_accuracy_kmeans, test_clustering_accuracy_kmeans,
                                knn_accuracy, logistic_accuracy])

        # save
        save_list(result_list,
                  save_path + dataset + '_depth_' + str(depth) + '_width_' + str(num_neuron) + '_iter' + str(iter + 1))