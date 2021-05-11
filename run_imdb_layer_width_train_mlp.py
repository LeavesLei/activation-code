from load_data import *
from activation_code_methods import *
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
import argparse
#from keras.utils import np_utils
from keras import backend as K
from keras.datasets import imdb
import numpy as np
from keras.utils import to_categorical
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Basic hyper-parameters
batch_size = 512
epoch = 20
repeat = 5
begin_repeat = 1
save_path = '/public/data1/users/leishiye/neural_code/models/layer_width/layer_width_'
depth = 1
dataset = 'imdb'

num_classes = 2
weight_decay = 1e-6
lr = 1e-2

width_list = [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

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

"""
num_train = int(x_train.shape[0] * 0.8)
num_val = x_train.shape[0] - num_train
mask = list(range(num_train, num_train+num_val))
x_val = x_train[mask]
y_val = y_train[mask]

mask = list(range(num_train))
x_train = x_train[mask]
y_train = y_train[mask]
"""
input_shape = x_train.shape[1:]

print('dataset: ' + dataset)
print('depth: ' + str(depth))

for iter in np.linspace(begin_repeat-1, begin_repeat + repeat-2, repeat).astype('int'):
    print('repeat: ' + str(iter + 1))
    for num_neuron in width_list:
        print('layer width: ' + str(num_neuron))
        mlp = create_mlp(num_neuron, depth, input_shape, num_classes, weight_decay=weight_decay, bn=False)

        # Compile networks
        opt = keras.optimizers.Adam(lr=lr)
        mlp.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        mlp.summary()

        # training networks
        mlp.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, verbose=1)

        # compute activation code
        train_activation_codes, test_activation_codes = compute_activation_code_for_mlp(x_train, x_test, model=mlp)

        # compute redundancy ratio
        test_redundancy_ratio = (test_activation_codes.shape[0] - np.unique(test_activation_codes, axis=0).shape[0]) / x_test.shape[0]
        train_redundancy_ratio = (train_activation_codes.shape[0] - np.unique(train_activation_codes, axis=0).shape[0]) / x_train.shape[0]

        print("train redundancy ratio: " + str(train_redundancy_ratio))
        print("test redundancy ratio: " + str(test_redundancy_ratio))

        mlp.save(save_path + str(num_neuron) + '_' + dataset + '_depth_' +str(depth)  + '_iter' + str(iter + 1) + '.h5')

