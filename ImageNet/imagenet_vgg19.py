import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
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

neural_code_dir = '/public/data1/users/leishiye/neural_code/results/neural_code_vgg19/vgg19_code_layer_'

for i in range(18):
    layer_code = np.load(neural_code_dir + str(i) + '.npy')
    # normalize
    layer_code = layer_code / (np.max(layer_code) - np.min(layer_code))
    if i == 0:
        neural_code = layer_code
    else:
        neural_code = np.hstack((neural_code, layer_code))

label_scalar = np.load('/public/data1/users/leishiye/neural_code/results/neural_code_vgg19/imagenet_val_label.npy')

# Split training data and test data
mask = np.array([i%5==0 for i in range(50000)])

train_activation_codes = neural_code[~mask]
train_label_scalar = label_scalar[~mask]

test_activation_codes = neural_code[mask]
test_label_scalar = label_scalar[mask]

print(train_activation_codes.shape)
print(train_label_scalar.shape)
print(test_activation_codes.shape)
print(test_label_scalar.shape)

n_clusters = 1000
cluster_result = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(neural_code)
clustering_accuracy_kmeans = compute_clustering_accuracy(cluster_result, label_scalar)

print("clustering_accuracy_kmeans: " + str(clustering_accuracy_kmeans))

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