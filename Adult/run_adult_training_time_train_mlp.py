import sys
sys.path.append("..") 
from load_data import *
from activation_code_methods import *
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from adult import Adult
import numpy as np
from keras.utils import to_categorical

# Basic hyper-parameters
batch_size = 128
repeat = 5
begin_repeat = 1
save_path = '/public/data1/users/leishiye/neural_code/models/training_time/model_training_process_'
depth = 1
dataset = 'adult'

num_classes = 2
weight_decay = 1e-6
lr = 1e-2

width_list = [60, 80, 100]
output_epoch_list = [1, 2, 3, 6, 10, 20, 30, 60, 100]

training_epoch_list = []
for i in range(len(output_epoch_list)):
    if i == 0:
        training_epoch_list.append(output_epoch_list[i] - 0)
    else:
        training_epoch_list.append(output_epoch_list[i] - output_epoch_list[i-1])

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

        # building model
        mlp = create_mlp(num_neuron, depth, input_shape, num_classes, weight_decay=weight_decay, bn=False)

        # Compile networks
        #opt = keras.optimizers.Adam(lr=lr)
        mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        mlp.summary()

        mlp.save(save_path + str(0) + '_width_' + str(num_neuron) + '_' + dataset + '_depth_' + str(depth) + '_iter' + str(iter + 1) + '.h5')

        # training according to training epoch list
        for index, training_epoch in enumerate(training_epoch_list):

            # training networks
            mlp.fit(x_train, y_train, batch_size=batch_size, epochs=training_epoch, verbose=1)

            mlp.save(save_path + str(output_epoch_list[index]) + '_width_' + str(num_neuron) + '_' + dataset + '_depth_' + str(depth) + '_iter' + str(iter + 1) + '.h5')