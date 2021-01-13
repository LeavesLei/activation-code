# encoding: utf-8
# @File  : run_layer_width_train_mlp.py
# @Author: LeavesLei
# @Date  : 2020/8/13

from load_data import *
from activation_code_methods import *
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
import argparse

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

num_classes = 10
weight_decay = 1e-6
lr = 1e-2

# Load data
width_list = [3, 7, 10, 15, 20, 23, 27, 30, 33, 37, 40, 43, 47, 50, 53, 57, 60, 65, 70, 75, 80, 90, 100]

if dataset == "cifar10":
    (x_train, y_train), (x_test, y_test) = load_cifar10(args.cifar10_path)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    width_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
                  550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
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

