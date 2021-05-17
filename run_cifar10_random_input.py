from load_data import *
from activation_code_methods import *
from keras.models import load_model

cifar10_path = '/public/data1/users/leishiye/datasets/cifar-10-batches-py'
repeat = 10
begin_repeat = 1
load_path = '/public/data1/users/leishiye/neural_code/models/training_time_cifar10/model_training_process_'
save_path = '/public/data1/users/leishiye/neural_code/results/training_time/result_list_training_process_'
depth = 5

dataset = 'cifar10'
num_classes = 10
n_clusters = num_classes

width_list = [50, 100, 200, 400]
output_epoch_list = [0, 1, 3, 6, 10, 20, 30, 40, 60, 80, 100, 120, 140, 160, 180, 200]

# load dataset
(x_train, y_train), (x_test, y_test) = load_cifar10(cifar10_path)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

num_train = int(x_train.shape[0] * 0.8)
num_val = x_train.shape[0] - num_train
mask = list(range(num_train, num_train+num_val))
x_val = x_train[mask]
y_val = y_train[mask]

mask = list(range(num_train))
x_train = x_train[mask]
y_train = y_train[mask]

# Generate random instance
random_x_train = -np.random.rand(*x_train.shape)
random_x_test = -np.random.rand(*x_test.shape)

train_label_scalar = np.argmax(y_train, axis=1).squeeze()
val_label_scalar = np.argmax(y_val, axis=1).squeeze()
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

            """
            # evaluation
            train_score = mlp.evaluate(x_train, y_train, verbose=0)
            print("train loss: ", train_score[0])
            print("train accuracy: ", train_score[1])
            test_score = mlp.evaluate(x_test, y_test, verbose=0)
            print("test loss: ", test_score[0])
            print("test accuracy: ", test_score[1])
            """
            # compute activation code
            train_activation_codes, test_activation_codes = compute_activation_code_for_mlp(random_x_train, random_x_test, model=mlp)

            # compute redundancy ratio
            test_redundancy_ratio = (test_activation_codes.shape[0] - np.unique(test_activation_codes, axis=0).shape[
                0]) / x_test.shape[0]
            train_redundancy_ratio = (train_activation_codes.shape[0] - np.unique(train_activation_codes, axis=0).shape[
                0]) / x_train.shape[0]

            print("train redundancy ratio: " + str(train_redundancy_ratio))
            print("test redundancy ratio: " + str(test_redundancy_ratio))

            """
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
            logistic_classifier = OneVsOneClassifier(LogisticRegression(random_state=9)).fit(train_activation_codes,
                                                                                             train_label_scalar)
            logistic_pred_result = logistic_classifier.predict(test_activation_codes)
            smstr = np.nonzero(test_label_scalar - logistic_pred_result)
            logistic_accuracy = 1 - np.shape(smstr[0])[0] / test_label_scalar.shape[0]

            print("logistic_accuracy: " + str(logistic_accuracy))
            """
            result_list.extend([train_redundancy_ratio, test_redundancy_ratio])

        # save
        save_list(result_list,
                  save_path + dataset + '_random_input_depth_' + str(depth) + '_width_' + str(num_neuron) + '_iter' + str(iter + 1))