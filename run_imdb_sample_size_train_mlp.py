# encoding: utf-8
# @File  : run_sample_size_train_mlp.py
# @Author: LeavesLei
# @Date  : 2020/8/13

from load_data import *
from activation_code_methods import *
import numpy as np
import argparse
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.datasets import imdb
import numpy as np
from keras.utils import to_categorical

# Basic hyper-parameters
batch_size = 512
epoch = 20
repeat = 5
begin_repeat = 1
save_path = '/public/data1/users/leishiye/neural_code/models/sample_size/model_sample_size_'
depth = 1
dataset = 'imdb'

num_classes = 2

num_classes = 10
weight_decay = 1e-6
lr = 1e-2

width_list = [100]

sample_size_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 15000, 25000]
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

            # building model
            mlp = create_mlp(num_neuron, depth, input_shape, num_classes, weight_decay=weight_decay, bn=True)

            # Compile networks
            #opt = keras.optimizers.Adam(lr=lr)
            mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            mlp.summary()

            # Training policies
            def lr_scheduler(epoch):
                return lr * (0.1 ** (epoch // 20))

            reduce_lr = LearningRateScheduler(lr_scheduler)
            early_stopping = EarlyStopping(monitor='loss', patience=10)

            # training set
            x_sub_train = x_train[:sample_size]
            y_sub_train = y_train[:sample_size]

            # extend the dataset
            expansion_factor = x_train.shape[0] // sample_size
            x_sub_train_expansion = np.tile(x_sub_train, (expansion_factor, 1))
            y_sub_train_expansion = np.tile(y_sub_train, (expansion_factor, 1))

            # training networks
            mlp.fit(x_sub_train_expansion, y_sub_train_expansion, batch_size=batch_size, epochs=epoch, verbose=1)

            # compute activation code
            train_activation_codes, test_activation_codes = compute_activation_code_for_mlp(x_train, x_test, model=mlp)

            # compute redundancy ratio
            test_redundancy_ratio = (test_activation_codes.shape[0] - np.unique(test_activation_codes, axis=0).shape[0]) / x_test.shape[0]
            train_redundancy_ratio = (train_activation_codes.shape[0] - np.unique(train_activation_codes, axis=0).shape[0]) / x_train.shape[0]

            print("train redundancy ratio: " + str(train_redundancy_ratio))
            print("test redundancy ratio: " + str(test_redundancy_ratio))

            mlp.save(save_path + str(sample_size) + '_width_' + str(num_neuron) + '_' + dataset + '_depth_' +
                     str(depth) + '_iter' + str(iter + 1) + '.h5')

        
