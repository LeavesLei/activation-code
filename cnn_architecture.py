# encoding: utf-8

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, concatenate, Lambda, add
from keras.models import Model
from keras.regularizers import l2
from keras.initializers import glorot_uniform
from keras import optimizers, regularizers

from keras.layers import Dropout
from keras.layers import MaxPool2D



# resnext parameters
weight_decay=5e-4

# densenet parameters
growth_rate = 3
depth = 30
compression = 0.5

def conv2d_bn_relu(model,
                   filters,
                   block_index, layer_index,
                   weight_decay=.0, padding='same'):
    conv_name = 'conv' + str(block_index) + '-' + str(layer_index)
    model = Conv2D(filters=filters,
                   kernel_size=(3, 3),
                   padding=padding,
                   kernel_regularizer=regularizers.l2(weight_decay),
                   strides=(1, 1),
                   name=conv_name,
                   )(model)
    bn_name = 'bn' + str(block_index) + '-' + str(layer_index)
    model = BatchNormalization(name=bn_name)(model)
    relu_name = 'relu' + str(block_index) + '-' + str(layer_index)
    model = Activation('relu', name=relu_name)(model)
    return model


def dense2d_bn_dropout(model, units, weight_decay, name):
    model = Dense(units,
                  kernel_regularizer=regularizers.l2(weight_decay),
                  name=name,
                  )(model)
    model = BatchNormalization(name=name + '-bn')(model)
    model = Activation('relu', name=name + '-relu')(model)
    model = Dropout(0.5, name=name + '-dropout')(model)
    return model


def conv2d_bn_in_resnet(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(weight_decay)
                   )(x)
    layer = BatchNormalization()(layer)
    return layer


def conv2d_bn_relu_in_resnet(x, filters, kernel_size, weight_decay=.0, strides=(1, 1)):
    layer = conv2d_bn_in_resnet(x, filters, kernel_size, weight_decay, strides)
    layer = Activation('relu')(layer)
    return layer

def VGG16(classes, input_shape, weight_decay,
           conv_block_num=5,
           fc_layers=2, fc_units=4096):
    input = Input(shape=input_shape)
    # block 1
    x = conv2d_bn_relu(model=input,
                       filters=64,
                       block_index=1, layer_index=1,
                       weight_decay=weight_decay
                       )
    x = conv2d_bn_relu(model=x,
                       filters=64,
                       block_index=1, layer_index=2,
                       weight_decay=weight_decay)
    x = MaxPool2D(name='pool1')(x)

    # block 2
    if conv_block_num >= 2:
        x = conv2d_bn_relu(x,
                           filters=128,
                           block_index=2, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=128,
                           block_index=2, layer_index=2,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool2')(x)

    # block 3
    if conv_block_num >= 3:
        x = conv2d_bn_relu(x,
                           filters=256,
                           block_index=3, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=256,
                           block_index=3, layer_index=2,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=256,
                           block_index=3, layer_index=3,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool3')(x)

    # block 4
    if conv_block_num >= 4:
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=4, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=4, layer_index=2,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=4, layer_index=3,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool4')(x)

    # block 5
    if conv_block_num >= 5:
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=5, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=5, layer_index=2,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=5, layer_index=3,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool5')(x)

    x = Flatten(name='flatten')(x)
    if fc_layers >= 1:
        x = dense2d_bn_dropout(x, fc_units, weight_decay, 'fc6')
        if fc_layers >= 2:
            x = dense2d_bn_dropout(x, fc_units, weight_decay, 'fc7')
    out = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(input, out)
    return model


def VGG19(classes, input_shape, weight_decay,
           conv_block_num=5,
           fc_layers=2, fc_units=4096):
    input = Input(shape=input_shape)
    # block 1
    x = conv2d_bn_relu(model=input,
                       filters=64,
                       block_index=1, layer_index=1,
                       weight_decay=weight_decay
                       )
    x = conv2d_bn_relu(model=x,
                       filters=64,
                       block_index=1, layer_index=2,
                       weight_decay=weight_decay)
    x = MaxPool2D(name='pool1')(x)

    # block 2
    if conv_block_num >= 2:
        x = conv2d_bn_relu(x,
                           filters=128,
                           block_index=2, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=128,
                           block_index=2, layer_index=2,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool2')(x)

    # block 3
    if conv_block_num >= 3:
        x = conv2d_bn_relu(x,
                           filters=256,
                           block_index=3, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=256,
                           block_index=3, layer_index=2,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=256,
                           block_index=3, layer_index=3,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=256,
                           block_index=3, layer_index=4,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool3')(x)

    # block 4
    if conv_block_num >= 4:
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=4, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=4, layer_index=2,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=4, layer_index=3,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=4, layer_index=4,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool4')(x)

    # block 5
    if conv_block_num >= 5:
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=5, layer_index=1,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=5, layer_index=2,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=5, layer_index=3,
                           weight_decay=weight_decay)
        x = conv2d_bn_relu(x,
                           filters=512,
                           block_index=5, layer_index=4,
                           weight_decay=weight_decay)
        x = MaxPool2D(name='pool5')(x)

    x = Flatten(name='flatten')(x)
    if fc_layers >= 1:
        x = dense2d_bn_dropout(x, fc_units, weight_decay, 'fc6')
        if fc_layers >= 2:
            x = dense2d_bn_dropout(x, fc_units, weight_decay, 'fc7')
    out = Dense(classes, activation='softmax', name='predictions')(x)
    model = Model(input, out)
    return model


def ResidualBlock(x, filters, kernel_size, weight_decay, downsample=True):
    if downsample:
        # residual_x = conv2d_bn_relu(x, filters, kernel_size=1, strides=2)
        residual_x = conv2d_bn_in_resnet(x, filters, kernel_size=1, strides=2)
        stride = 2
    else:
        residual_x = x
        stride = 1
    residual = conv2d_bn_relu_in_resnet(x,
                              filters=filters,
                              kernel_size=kernel_size,
                              weight_decay=weight_decay,
                              strides=stride,
                              )
    residual = conv2d_bn_in_resnet(residual,
                         filters=filters,
                         kernel_size=kernel_size,
                         weight_decay=weight_decay,
                         strides=1,
                         )
    out = layers.add([residual_x, residual])
    out = Activation('relu')(out)
    return out


def ResNet18(classes, input_shape, weight_decay=1e-4):
    input = Input(shape=input_shape)
    x = input
    # x = conv2d_bn_relu(x, filters=64, kernel_size=(7, 7), weight_decay=weight_decay, strides=(2, 2))
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2),  padding='same')(x)
    x = conv2d_bn_relu_in_resnet(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, strides=(1, 1))

    # # conv 2
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # # conv 3
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=128, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # # conv 4
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=256, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # # conv 5
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    x = ResidualBlock(x, filters=512, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    x = AveragePooling2D(pool_size=(4, 4), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(input, x, name='ResNet18')
    return model


def ResNetForCIFAR10(classes, name, input_shape, block_layers_num, weight_decay):
    input = Input(shape=input_shape)
    x = input
    x = conv2d_bn_relu_in_resnet(x, filters=16, kernel_size=(3, 3), weight_decay=weight_decay, strides=(1, 1))

    # # conv 2
    for i in range(block_layers_num):
        x = ResidualBlock(x, filters=16, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # # conv 3
    x = ResidualBlock(x, filters=32, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    for i in range(block_layers_num - 1):
        x = ResidualBlock(x, filters=32, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    # # conv 4
    x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=True)
    for i in range(block_layers_num - 1):
        x = ResidualBlock(x, filters=64, kernel_size=(3, 3), weight_decay=weight_decay, downsample=False)
    x = AveragePooling2D(pool_size=(8, 8), padding='valid')(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(input, x, name=name)
    return model


def ResNet20ForCIFAR10(classes, input_shape, weight_decay):
    return ResNetForCIFAR10(classes, 'resnet20', input_shape, 3, weight_decay)


def ResNet32ForCIFAR10(classes, input_shape, weight_decay):
    return ResNetForCIFAR10(classes, 'resnet32', input_shape, 5, weight_decay)


def ResNet56ForCIFAR10(classes, input_shape, weight_decay):
    return ResNetForCIFAR10(classes, 'resnet56', input_shape, 9, weight_decay)