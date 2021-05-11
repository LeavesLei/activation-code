from ..load_data import *
from ..activation_code_methods import *
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
import argparse
#from keras.utils import np_utils
from keras import backend as K
import numpy as np
from keras.utils import to_categorical
from ..adult import Adult
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Basic hyper-parameters
batch_size = 128
epoch = 100
repeat = 5
begin_repeat = 1
save_path = '/public/data1/users/leishiye/neural_code/models/layer_width/layer_width_'
depth = 1
dataset = 'adult'

num_classes = 2
weight_decay = 1e-6
lr = 1e-2

width_list = [10, 20, 40, 60, 80, 100, 150, 200, 250, 300]

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

input_shape = x_train.shape[1:]

print('dataset: ' + dataset)
print('depth: ' + str(depth))

for iter in np.linspace(begin_repeat-1, begin_repeat + repeat-2, repeat).astype('int'):
    print('repeat: ' + str(iter + 1))
    for num_neuron in width_list:
        print('layer width: ' + str(num_neuron))
        mlp = create_mlp(num_neuron, depth, input_shape, num_classes, weight_decay=weight_decay, bn=False)

        # Compile networks
        #opt = keras.optimizers.Adam(lr=lr)
        mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        mlp.summary()

        # training networks
        mlp.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, verbose=0)

        mlp.save(save_path + str(num_neuron) + '_' + dataset + '_depth_' +str(depth)  + '_iter' + str(iter + 1) + '.h5')

