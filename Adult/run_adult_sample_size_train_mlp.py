import sys
sys.path.append("..") 
from load_data import *
from activation_code_methods import *
import numpy as np
import argparse
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras import backend as K
from adult import Adult
import numpy as np
from keras.utils import to_categorical

# Basic hyper-parameters
batch_size = 128
epoch = 100
repeat = 3
begin_repeat = 3
save_path = '/public/data1/users/leishiye/neural_code/models/sample_size/model_sample_size_'
depth = 1
dataset = 'adult'

num_classes = 2
weight_decay = 1e-6
lr = 1e-2

width_list = [60, 80, 100]

sample_size_list = [100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000]
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
        result_list = []
        for sample_size in sample_size_list:
            print('sample size: ' + str(sample_size))

            # building model
            mlp = create_mlp(num_neuron, depth, input_shape, num_classes, weight_decay=weight_decay, bn=False)

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

            mlp.save(save_path + str(sample_size) + '_width_' + str(num_neuron) + '_' + dataset + '_depth_' +
                     str(depth) + '_iter' + str(iter + 1) + '.h5')

        
