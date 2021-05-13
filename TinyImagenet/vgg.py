import torch.nn as tnn

def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer

def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [ conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list)) ]
    layers += [ tnn.MaxPool2d(kernel_size = pooling_k, stride = pooling_s)]
    return tnn.Sequential(*layers)

def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        #tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer

class VGG16(tnn.Module):
    def __init__(self, n_classes=10, input_channel=3, layer_width=64):
        super(VGG16, self).__init__()

        self.layer1 = conv_layer(input_channel, 64, 3, 1)
        self.layer2 = conv_layer(64, 64, 3, 1)
        
        self.layer3 = conv_layer(64, 128, 3, 1)
        self.layer4 = conv_layer(128, 128, 3, 1)

        self.layer5 = conv_layer(128, 256, 3, 1)
        self.layer6 = conv_layer(256, 256, 3, 1)
        self.layer7 = conv_layer(256, 256, 3, 1)

        self.layer8 = conv_layer(256, 512, 3, 1)
        self.layer9 = conv_layer(512, 512, 3, 1)
        self.layer10 = conv_layer(512, 512, 3, 1)

        self.layer11 = conv_layer(512, 512, 3, 1)
        self.layer12 = conv_layer(512, 512, 3, 1)
        self.layer13 = conv_layer(512, 512, 3, 1)

        # FC layers
        self.layer14 = vgg_fc_layer(512*4, layer_width)
        self.layer15 = vgg_fc_layer(layer_width, layer_width)

        # Final layer
        self.layer16 = tnn.Linear(layer_width, n_classes)

        self.maxpool = tnn.MaxPool2d(2,2)

    def forward(self, x):
        out = self.layer1(x)
        feature1 = out
        out = self.layer2(out)
        feature2 = out
        out = self.maxpool(out)

        out = self.layer3(out)
        feature3 = out
        out = self.layer4(out)
        feature4 = out
        out = self.maxpool(out)

        out = self.layer5(out)
        feature5 = out
        out = self.layer6(out)
        feature6 = out
        out = self.layer7(out)
        feature7 = out
        out = self.maxpool(out)

        out = self.layer8(out)
        feature8 = out
        out = self.layer9(out)
        feature9 = out
        out = self.layer10(out)
        feature10 = out
        out = self.maxpool(out)

        out = self.layer11(out)
        feature11 = out
        out = self.layer12(out)
        feature12 = out
        out = self.layer13(out)
        feature13 = out
        out = self.maxpool(out)

        vgg16_features = out
        out = vgg16_features.view(out.size(0), -1)
        #print(out.shape)
        out = self.layer14(out)
        feature14 = out
        out = self.layer15(out)
        feature15 = out
        out = self.layer16(out)

        return feature14, feature15, out