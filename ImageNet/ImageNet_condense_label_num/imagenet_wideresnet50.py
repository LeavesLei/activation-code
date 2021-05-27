import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_class', type=int, default=1000, help='number of classes')
args = parser.parse_args()

num_class = args.num_class

# Load neural code

def compute_clustering_accuracy(cluster_result, label, n_cluster=1000):
    partition_matrix = contingency_matrix(label, cluster_result)
    _, label_mapping = linear_sum_assignment(-partition_matrix)
    for i in range(n_cluster):
        cluster_result[cluster_result == label_mapping[i]] = n_cluster + i
    cluster_result = cluster_result - n_cluster
    smstr = np.nonzero(label - cluster_result)
    clustering_accuracy = 1 - np.shape(smstr[0])[0] / label.shape[0]
    return clustering_accuracy

neural_code_dir = '/export/leishiye/neural_code/neural_code_wideresnet50/wideresnet50_code_layer_'

for i in range(17):
    layer_code = np.load(neural_code_dir + str(i) + '.npy')
    # normalize
    layer_code = layer_code / (np.max(layer_code) - np.min(layer_code))
    if i == 0:
        neural_code = layer_code
    else:
        neural_code = np.hstack((neural_code, layer_code))

label_scalar = np.load('/export/leishiye/neural_code/neural_code_wideresnet50/imagenet_val_label.npy')

# Condense neural code and label
ratio = 1000 // num_class
neural_code = neural_code[label_scalar % ratio == 0]
label_scalar = label_scalar[label_scalar % ratio == 0]
label_scalar = label_scalar // ratio

# Split training data and test data
mask = np.array([i%5==0 for i in range(num_class*50)])

train_activation_codes = neural_code[~mask]
train_label_scalar = label_scalar[~mask]

test_activation_codes = neural_code[mask]
test_label_scalar = label_scalar[mask]

print(train_activation_codes.shape)
print(train_label_scalar.shape)
print(test_activation_codes.shape)
print(test_label_scalar.shape)

# compute multiclass logisticRegression
logistic_classifier = OneVsOneClassifier(LogisticRegression(solver='liblinear', random_state=9)).fit(train_activation_codes,
                                                                                    train_label_scalar)
logistic_pred_result = logistic_classifier.predict(test_activation_codes)
smstr = np.nonzero(test_label_scalar - logistic_pred_result)
logistic_accuracy = 1 - np.shape(smstr[0])[0] / test_label_scalar.shape[0]

print("logistic_accuracy: " + str(logistic_accuracy))

# compute clusterisng accuracy with KNN
neigh = KNeighborsClassifier(n_neighbors=9, metric='euclidean').fit(train_activation_codes,
                                                                    train_label_scalar)
knn_pred_result = neigh.predict(test_activation_codes)
smstr = np.nonzero(test_label_scalar - knn_pred_result)
knn_accuracy = 1 - np.shape(smstr[0])[0] / test_label_scalar.shape[0]

print("knn_accuracy: " + str(knn_accuracy))

n_clusters = num_class
cluster_result = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(neural_code)
clustering_accuracy_kmeans = compute_clustering_accuracy(cluster_result, label_scalar, n_cluster=num_class)

print("clustering_accuracy_kmeans: " + str(clustering_accuracy_kmeans))

np.save('/export/leishiye/results/wideresnet50_results_num_class' + str(num_class), np.array([clustering_accuracy_kmeans, knn_accuracy, logistic_accuracy]))