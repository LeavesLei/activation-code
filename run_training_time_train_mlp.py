# encoding: utf-8
# @File  : run.py
# @Author: LeavesLei
# @Date  : 2020/8/13

from load_data import *
from activation_code_methods import *
from keras.callbacks import LearningRateScheduler
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--begin_repeat', type=int, default=1, help=' begin repeat num')
parser.add_argument('--repeat', type=int, default=2, help='repeat times')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--mnist_path', type=str, default='mnist.npz', help='MNIST path')
parser.add_argument('--cifar10_path', type=str, default='cifar-10-batches-py', help='CIFAR10 path')
parser.add_argument('--save_path', type=str, default='model/training_time/model_training_process_', help='save path')
parser.add_argument('--depth', type=int, default=5, help='depth')
args = parser.parse_args()

batch_size = args.batch_size
repeat = args.repeat
begin_repeat = args.begin_repeat
save_path = args.save_path
depth = args.depth

dataset = args.dataset
num_classes = 10
weight_decay = 1e-6
lr = 1e-2

width_list = [40, 50, 60]
output_epoch_list = [1, 3, 6, 10, 20, 30, 40, 60, 80, 100, 120, 140, 160, 180, 200]

training_epoch_list = []
for i in range(len(output_epoch_list)):
    if i == 0:
        training_epoch_list.append(output_epoch_list[i] - 0)
    else:
        training_epoch_list.append(output_epoch_list[i] - output_epoch_list[i-1])

if dataset == "cifar10":
    (x_train, y_train), (x_test, y_test) = load_cifar10(args.cifar10_path)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    width_list = [50, 100, 200, 400]
# laod MNIST
elif dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = load_mnist(path=args.mnist_path, flatten=True)

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
        result_list = []

        # building model
        mlp = create_mlp(num_neuron, depth, input_shape, num_classes, weight_decay=weight_decay, bn=True)

        # Compile networks
        opt = keras.optimizers.Adam(lr=lr)
        mlp.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        mlp.summary()

        mlp.save(save_path + str(0) + '_width_' + str(num_neuron) + '_' + dataset + '_depth_' + str(depth) + '_iter' + str(iter + 1) + '.h5')

        # training according to training epoch list
        for index, training_epoch in enumerate(training_epoch_list):
            # Training policies
            if index == 0:
                def lr_scheduler(epoch):
                    return lr * (0.1 ** (epoch // 20))
            else:
                def lr_scheduler(epoch):
                    return lr * (0.1 ** ((output_epoch_list[index-1] + epoch) // 20))
            reduce_lr = LearningRateScheduler(lr_scheduler)

            # training networks
            mlp.fit(x_train, y_train, batch_size=batch_size, epochs=training_epoch,
                    callbacks=[reduce_lr], validation_data=(x_val, y_val), verbose=1)

            mlp.save(save_path + str(output_epoch_list[index]) + '_width_' + str(num_neuron) + '_' + dataset + '_depth_' + str(depth) + '_iter' + str(iter + 1) + '.h5')