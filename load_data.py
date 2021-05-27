# encoding: utf-8
# @File  : load_data.py
# @Author: LeavesLei
# @Date  : 2020/8/13

import os
import numpy as np
import pickle

import keras.backend as K
from keras.utils import np_utils


def one_hot(x, n):
    """
    convert index representation to one-hot representation
    """
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]


def load_mnist(path='/public/data1/users/leishiye/datasets/mnist.npz', flatten=False):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    # Adapt the data as an input of a fully-connected (flatten to 1D)
    if flatten:
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)

    # Normalize data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train / 255
    x_test = x_test / 255

    # Adapt the labels to the one-hot vector syntax required by the softmax
    from keras.utils import np_utils
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


# 对每一个batch进行导入
def _load_batch(file):
    with open(file, 'rb') as fo:  # 打开文件
        d = pickle.load(fo, encoding='bytes')  # 导入文件，以bytes编码格式
        d_decoded = {}  # 定义字典
        for k, v in d.items():  # 对数据集中的特征与标签遍历
            d_decoded[k.decode('utf8')] = v  # 用数据集填充d_decoded
        d = d_decoded
        data = d['data']  # 读出batch中所有特征存入data
        labels = d['labels']  # 读出batch中所有标签存入labels
        data = data.reshape(data.shape[0], 3, 32, 32)  # 数组形变
    return data, labels  # 返回最原始的特征与标签


def load_cifar10(path='/public/data1/users/leishiye/datasets/cifar-10-batches-py'):
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    # 这里导入为了判断图像格式约定
    # from tensorflow.keras import backend as K

    # 训练集训练数据数量
    num_train_samples = 50000

    # 新建数组变量，根据数据集情况建立，50000张3通道的32*32的图片
    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    # 这里一共有5组data_batch，1到5循环遍历，每一个batch有10000张图片，导入图片
    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))  # 根据文件夹进行绝对地址访问
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],  # 1-10000，10001-20000，20001-30000 等等    3通道32*32的彩色图
         y_train[(i - 1) * 10000: i * 10000]) = _load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = _load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))  # 增加一维,相当于把行向量变成列向量
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':  # image_data_format返回默认图像格式约定，either 'channels_first' or 'channels_last'
        x_train = x_train.transpose(0, 2, 3, 1)  # 轴变换，按照0,2,3,1进行
        x_test = x_test.transpose(0, 2, 3, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)



def _grayscale(a):
    print(a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1))
    return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1)


def _load_batch_cifar100(path="/home1/leishiye/dataset/cifar-100-python", dtype='float64', containing_channel=True):
    """
    load a batch in the CIFAR-100 format
    """
    batch = np.load(path, encoding='bytes', allow_pickle=True)
    data = batch[b'data'] / 255.0
    labels = one_hot(batch[b'fine_labels'], n=100)
    if containing_channel:
        data = data.reshape(data.shape[0], 3, 32, 32).transpose((0, 2, 3, 1))
    return data.astype(dtype), labels.astype(dtype)


def load_cifar100(path="/home1/leishiye/dataset/cifar-100-python", dtype='float64', grayscale=False, containing_channel=True):
    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")
    x_train, t_train = _load_batch_cifar100(train_path, dtype=dtype, containing_channel=containing_channel)
    x_test, t_test = _load_batch_cifar100(test_path, dtype=dtype, containing_channel=containing_channel)

    if grayscale:
        x_train = _grayscale(x_train)
        x_test = _grayscale(x_test)
    return x_train, t_train, x_test, t_test