from load_data import *
from activation_code_methods import *
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
import argparse
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
#from keras.utils import np_utils
from keras import backend as K

import numpy as np

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epoch', type=int, default=200, help='epoch')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--begin_repeat', type=int, default=1, help=' begin repeat num')
parser.add_argument('--repeat', type=int, default=2, help='repeat times')
parser.add_argument('--mnist_path', type=str, default='mnist.npz', help='MNIST path')
parser.add_argument('--cifar10_path', type=str, default='cifar-10-batches-py', help='CIFAR10 path')
parser.add_argument('--save_path', type=str, default='model/layer_width/model_layer_width_', help='save path')
parser.add_argument('--depth', type=int, default=5, help='depth')
args = parser.parse_args()

# Basic hyper-parameters
batch_size = args.batch_size
epoch = args.epoch
repeat = args.repeat
begin_repeat = args.begin_repeat
save_path = args.save_path
depth = args.depth
dataset = args.dataset

num_classes = 2
weight_decay = 1e-6
lr = 1e-2

# Load data
width_list = [3, 7, 10, 15, 20, 23, 27, 30, 33, 37, 40, 43, 47, 50, 53, 57, 60, 65, 70, 75, 80, 90, 100]

##########################################
from keras.datasets import imdb

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
X_train = vectorize_sequences(x_train)
X_test = vectorize_sequences(x_test)

# Convert labels from integers to floats:
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')
###########################################


num_train = int(x_train.shape[0] * 0.8)
num_val = x_train.shape[0] - num_train
mask = list(range(num_train, num_train+num_val))
x_val = x_train[mask]
y_val = y_train[mask]

mask = list(range(num_train))
x_train = x_train[mask]
y_train = y_train[mask]

input_shape = x_train.shape[1:]

print('dataset: ' + dataset)
print('depth: ' + str(depth))

for iter in np.linspace(begin_repeat-1, begin_repeat + repeat-2, repeat).astype('int'):
    print('repeat: ' + str(iter + 1))
    for num_neuron in width_list:
        print('layer width: ' + str(num_neuron))
        mlp = create_mlp(num_neuron, depth, input_shape, num_classes, weight_decay=weight_decay, bn=True)

        # Compile networks
        opt = keras.optimizers.Adam(lr=lr)
        mlp.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        mlp.summary()

        # Training policies
        def lr_scheduler(epoch):
            return lr * (0.1 ** (epoch // 20))

        reduce_lr = LearningRateScheduler(lr_scheduler)
        early_stopping = EarlyStopping(monitor='loss', patience=10)

        # training networks
        mlp.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, callbacks=[reduce_lr, early_stopping],
                validation_data=(x_val, y_val), verbose=1)

        mlp.save(save_path + str(num_neuron) + '_' + dataset + '_depth_' +str(depth)  + '_iter' + str(iter + 1) + '.h5')

