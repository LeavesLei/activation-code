# encoding: utf-8
# @File  : activation_code_methods.py
# @Author: LeavesLei
# @Date  : 2020/8/13

import numpy as np
import keras
import keras.backend as K

from numpy import random
from numpy import linalg

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras import regularizers



def create_mlp(num_neuron, depth, input_shape, num_classes=10, weight_decay=0., bn=False):
    mlp =  Sequential()
    for i in range(depth):
        mlp.add(Dense(num_neuron // depth, input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
        if bn:
            mlp.add(BatchNormalization())
        mlp.add(Activation('relu'))
    mlp.add(Dense(num_classes, activation='softmax'))
    return mlp

def compute_activation_code_for_mlp(training_samples, test_samples, model):
    samples = np.vstack((training_samples, test_samples))
    inp = model.input  # input placeholder
    activation_layers = [x for x in model.layers if
                         isinstance(x, keras.layers.core.Activation)]  # all activation layers
    outputs = [layer.output for layer in activation_layers]  # all activation layer outputs
    functor = K.function([inp], outputs)  # evaluation function

    layer_outs = functor([samples])
    layer_outs = np.array(layer_outs).swapaxes(1, 0)
    layer_outs = layer_outs.reshape((len(layer_outs), -1))
    layer_outs = np.where(layer_outs>0, 1, 0).astype(np.int8)
    for i in [0, 1]:
        idx = np.argwhere(np.all(layer_outs[..., :] == i, axis=0))
        layer_outs = np.delete(layer_outs, idx, axis=1)
    return np.vsplit(layer_outs, [training_samples.shape[0]])


def compute_activation_code_for_cnn(training_samples, test_samples, model):
    samples = np.vstack((training_samples, test_samples))
    inp = model.input  # input placeholder
    activation_layers = [x for x in model.layers if
                         isinstance(x, keras.layers.core.Activation)]  # all activation layers
    outputs = [layer.output for layer in activation_layers]  # all activation layer outputs
    functor = K.function([inp], outputs)  # evaluation function

    for index in range(samples.shape[0] // 1000):
        if index % 5 == 0:
            print("load activation code index: " + str(index))
        sub_layer_outs = functor([samples[index * 1000: index * 1000 + 1000]])
        for i in range(len(sub_layer_outs)):
            sub_layer_outs[i] = sub_layer_outs[i].reshape(len(sub_layer_outs[i]), -1)
            sub_layer_outs[i] = np.where(sub_layer_outs[i] > 0, 1, 0).astype(np.int8)
        sub_layer_outs = np.hstack(sub_layer_outs)
        if index == 0:
            layer_outs = sub_layer_outs
        else:
            layer_outs = np.vstack((layer_outs, sub_layer_outs))
    for i in [0, 1]:
        idx = np.argwhere(np.all(layer_outs[..., :] == i, axis=0))
        layer_outs = np.delete(layer_outs, idx, axis=1)
    return np.vsplit(layer_outs, [training_samples.shape[0]])


# creat noise samples
def create_noise_samples(training_samples, training_labels, label_noise_ratio=0):
    # training_labels为one-hot形式
    noise_sample_num = int(label_noise_ratio * len(training_samples))
    np.random.seed = 10
    index = [i for i in range(len(training_samples))]
    index = random.shuffle(index)
    samples = np.squeeze(training_samples[index])
    labels = np.squeeze(training_labels[index])

    noise_samples = samples[: noise_sample_num]
    true_samples = samples[noise_sample_num:]

    noise_labels = labels[: noise_sample_num]
    true_labels = labels[noise_sample_num:]

    np.random.seed = 7
    random.shuffle(noise_labels)
    samples = np.vstack((true_samples, noise_samples))
    labels = np.vstack((true_labels, noise_labels))

    index = [i for i in range(len(samples))]
    index = random.shuffle(index)
    samples = samples[index]
    labels = labels[index]

    samples = np.squeeze(samples)
    labels = np.squeeze(labels)

    return samples, labels


def create_noise_instances(training_samples, training_labels, instance_noise_ratio=0):
    noise_instance_num = int(instance_noise_ratio * len(training_samples))
    np.random.seed = 10
    x_train_random = np.random.rand(training_samples.shape[0], training_samples.shape[1])

    noise_samples = x_train_random[: noise_instance_num]
    true_samples = training_samples[noise_instance_num:]
    samples = np.vstack((noise_samples, true_samples))

    index = [i for i in range(len(samples))]
    index = random.shuffle(index)
    samples = samples[index]
    labels = training_labels[index]

    samples = np.squeeze(samples)
    labels = np.squeeze(labels)

    return samples, labels


def save_list(file_name, path):
    file = open(path, 'w')
    for fp in file_name:
        file.write(str(fp))
        file.write('\n')
    file.close()


def load_list(path):
    data = []
    file_handler =open(path, mode='r')
    contents = file_handler.readlines()
    for name in contents:
        name = name.strip('\n')
        data.append(float(name))
    return data


def samples_of_trajectroy(sample, center):
    sample = np.squeeze(sample)
    diameter_vector = center - sample
    diameter_vector = diameter_vector / linalg.norm(diameter_vector)
    trajectory = np.ones((200, sample.shape[0]))
    for i, alpha in enumerate(np.hstack((np.linspace(-5, -0.05, 100), np.linspace(0.05, 5, 100)))):
        trajectory[i] = sample + alpha * diameter_vector
    return trajectory


def compute_diameter_of_linear_regions(sample, center, model):
    sample = sample.reshape(1, -1)
    src_activation_code = compute_activation_code_for_one_hidden_layer_network(samples=sample, model=model)
    trajectory = samples_of_trajectroy(sample, center)
    trajectory_activation_code = compute_activation_code_for_one_hidden_layer_network(samples=trajectory, model=model)
    distance = trajectory_activation_code - src_activation_code
    mask = np.all(np.equal(distance, 0), axis=1)
    diameter = distance[mask].shape[0]
    return diameter * 0.05


def compute_diameter_of_linear_regions_for_cnn(sample, center, model):
    #sample = sample.reshape(1, 32, 32, 3)
    sample = sample.reshape(1, -1)
    src_activation_code = compute_activation_code_for_cnn(samples=sample, model=model)
    trajectory = samples_of_trajectroy(sample, center)
    trajectory_activation_code = compute_activation_code_for_cnn(samples=trajectory, model=model)
    distance = trajectory_activation_code - src_activation_code
    mask = np.all(np.equal(distance, 0), axis=1)
    diameter = distance[mask].shape[0]
    return diameter * 0.05

def compute_clustering_accuracy(cluster_result, label):
    for i in range(10):
        single_result = []
        for j in range(10):
            result_copy = cluster_result.copy()
            result_copy[result_copy == i] = j
            smstr = np.nonzero(label - result_copy)
            single_result.append(np.shape(smstr[0])[0])
        cluster_result[cluster_result == i] = 10 + np.argmin(single_result)
    cluster_result = cluster_result - 10
    smstr = np.nonzero(label - cluster_result)
    clustering_accuracy = 1 - np.shape(smstr[0])[0] / label.shape[0]
    return clustering_accuracy