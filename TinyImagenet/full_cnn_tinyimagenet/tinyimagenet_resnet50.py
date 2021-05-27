import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_class', type=int, default=200, help='number of classes')
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

test_neural_code_dir = '/export/leishiye/neural_code_tinyimagenet/neural_code_resnet50/resnet50_test_code_layer_'
train_neural_code_dir = '/export/leishiye/neural_code_tinyimagenet/neural_code_resnet50/resnet50_train_code_layer_'

for i in range(17):
    test_layer_code = np.load(test_neural_code_dir + str(i) + '.npy')
    train_layer_code = np.load(train_neural_code_dir + str(i) + '.npy')
    # normalize
    test_layer_code = test_layer_code / (np.max(test_layer_code) - np.min(test_layer_code))
    train_layer_code = train_layer_code / (np.max(train_layer_code) - np.min(train_layer_code))
    if i == 0:
        test_activation_codes = test_layer_code
        train_activation_codes = train_layer_code
    else:
        test_activation_codes = np.hstack((test_activation_codes, test_layer_code))
        train_activation_codes = np.hstack((train_activation_codes, train_layer_code))

test_label_scalar = np.load('/export/leishiye/neural_code_tinyimagenet/neural_code_resnet50/test_label.npy')
train_label_scalar = np.load('/export/leishiye/neural_code_tinyimagenet/neural_code_resnet50/train_label.npy')

ratio = 200 // num_class
test_activation_codes = test_activation_codes[test_label_scalar % ratio == 0]
test_label_scalar = test_label_scalar[test_label_scalar % ratio == 0]
test_label_scalar = test_label_scalar // ratio

train_activation_codes = train_activation_codes[train_label_scalar % ratio == 0]
train_label_scalar = train_label_scalar[train_label_scalar % ratio == 0]
train_label_scalar = train_label_scalar // ratio

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
cluster_result = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(test_activation_codes)
test_clustering_accuracy_kmeans = compute_clustering_accuracy(cluster_result, test_label_scalar, n_cluster=num_class)

print("test_accuracy_kmeans: " + str(test_clustering_accuracy_kmeans))

cluster_result = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(train_activation_codes)
train_clustering_accuracy_kmeans = compute_clustering_accuracy(cluster_result, train_label_scalar, n_cluster=num_class)

print("train_accuracy_kmeans: " + str(train_clustering_accuracy_kmeans))

np.save('/export/leishiye/results/resnet50_tinyimagenet_results_num_class' + str(num_class), np.array([test_clustering_accuracy_kmeans, train_clustering_accuracy_kmeans, knn_accuracy, logistic_accuracy]))