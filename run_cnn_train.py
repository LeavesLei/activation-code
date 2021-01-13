#-*- coding:utf-8 _*-  
""" 
@author:Leaves
@file: run_cnn.py
@time: 2020/09/09
"""
import argparse
import gc

from activation_code_methods import *
from load_data import *
from cnn_architecture import *

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping


parser = argparse.ArgumentParser()
parser.add_argument('--architecture', type=str, default='vgg16', help='network architecture')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epoch', type=int, default=200, help='epoch')
parser.add_argument('--mnist_path', type=str, default='mnist.npz', help='MNIST path')
parser.add_argument('--cifar10_path', type=str, default='cifar-10-batches-py', help='CIFAR10 path')
parser.add_argument('--save_path', type=str, default='model/cnn/model_', help='save path')
args = parser.parse_args()

network_architecture = args.architecture
dataset = args.dataset
batch_size = args.batch_size
epoch = args.epoch
save_path = args.save_path

weight_decay = 5e-6
lr = 1e-2
num_classes = 10
data_augmentation = True

if dataset == "cifar10":
    (x_train, y_train), (x_test, y_test) = load_cifar10(args.cifar10_path)
    x_train = x_train.reshape(50000, 32, 32, 3)
    x_test = x_test.reshape(10000, 32, 32, 3)
elif dataset == "mnist":
    (x_train, y_train), (x_test, y_test) = load_mnist(path=args.mnist_path, flatten=False)
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)

num_train = int(x_train.shape[0] * 0.8)
num_val = x_train.shape[0] - num_train
mask = list(range(num_train, num_train+num_val))
x_val = x_train[mask]
y_val = y_train[mask]

mask = list(range(num_train))
x_train = x_train[mask]
y_train = y_train[mask]
print(x_train.shape)

# generate non-one-hot label for clustering
train_label_scalar = np.argmax(y_train, axis=1).squeeze()
test_label_scalar = np.argmax(y_test, axis=1).squeeze()

input_shape = x_train.shape[1:]

if network_architecture == 'vgg16':
    model = VGG16(classes=num_classes,
             input_shape=input_shape,
             weight_decay=weight_decay,
             conv_block_num=5,
             fc_layers=2,
             fc_units=512
             )
elif network_architecture == 'vgg19':
    model = VGG19(classes=num_classes,
             input_shape=input_shape,
             weight_decay=weight_decay,
             conv_block_num=5,
             fc_layers=2,
             fc_units=512
             )
elif network_architecture == 'resnet18':
    #weight_decay = 1e-4
    model = ResNet18(input_shape=input_shape, classes=num_classes, weight_decay=weight_decay)
elif network_architecture == 'resnet20':
    #weight_decay = 1e-4
    model = ResNet20ForCIFAR10(input_shape=input_shape, classes=num_classes, weight_decay=weight_decay)
elif network_architecture == 'resnet32':
    #weight_decay = 1e-4
    model = ResNet32ForCIFAR10(input_shape=input_shape, classes=num_classes, weight_decay=weight_decay)
elif network_architecture == 'resnet56':
    #weight_decay = 1e-4
    model = ResNet56ForCIFAR10(input_shape=input_shape, classes=num_classes, weight_decay=weight_decay)

print('architecture: ' + network_architecture)
print('dataset: ' + dataset)
print('input shape: ' + str(input_shape))
# Compile the network

#opt = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=False)
opt = keras.optimizers.Adam(lr=lr)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

def lr_scheduler(epoch):
    return lr * (0.1 ** (epoch // 50))

reduce_lr = LearningRateScheduler(lr_scheduler)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
#                               patience=10, min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='loss', patience=10)

if data_augmentation:
    # datagen
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False, # randomly flip images
    )
    # (std, mean, and principal components if ZCA whitening is applied).
    # datagen.fit(x_train)
    print('train with data augmentation')
    history = model.fit_generator(generator=datagen.flow(x_train, y_train, batch_size=batch_size),
                                epochs=epoch,
                                callbacks=[reduce_lr, early_stopping],
                                validation_data=(x_val, y_val)
                                )
else:
    print('train without data augmentation')
    history = model.fit(x_train, y_train,
                      batch_size=batch_size, epochs=epoch,
                      callbacks=[reduce_lr],
                      validation_data=(x_val, y_val)
                      )

model.save(save_path + network_architecture + '_' + dataset + '_adam' + '.h5')
