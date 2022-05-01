"""Convolutional head before RNN.

This file defines CNN head before RNN in combination of CNN+RNN.
"""
# import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import _image_standardization


# ResBlock， implement the ResBlock
class ResBlock(nn.Module):
    """This class defines the residual block."""

    def __init__(self, in_channel, out_channel, stride=1):
        """Initialize the object."""
        super(ResBlock, self).__init__()
        self.normal = nn.Sequential(
            nn.Conv2d(in_channel,
                      out_channel,
                      kernel_size=3,
                      stride=stride,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,
                      out_channel,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel,
                          out_channel,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        """Define forward process of residual block."""
        out = self.normal(x) + self.shortcut(x)
        out = F.relu(out)
        return out


class ConvolutionHead_Nvidia(nn.Module):
    """This class defines Nvidia CNN head."""

    def __init__(self, img_dim, time_sequence,
                 num_filters=8, features_per_filter=4):
        """Initialize the object."""
        super(ConvolutionHead_Nvidia, self).__init__()

        self.feature_layer = None
        self.filter_output = []
        self.linear = []
        self.num_filters = num_filters
        self.features_per_filter = features_per_filter
        self.conv = nn.Sequential(  # (66, 200)
            nn.Conv2d(3, 24, kernel_size=5, stride=2, bias=True),
            nn.ReLU(inplace=True),  # after (31,98)

            nn.Conv2d(24, 36, kernel_size=5, stride=2, bias=True),
            nn.ReLU(inplace=True),  # after (14,47)

            nn.Conv2d(36, 48, kernel_size=5, stride=2, bias=True),
            nn.ReLU(inplace=True),  # after (5,22)

            nn.Conv2d(48, 64, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(64),   # after (3,20)
            nn.ReLU(inplace=True),

            nn.Conv2d(64, self.num_filters,
                      kernel_size=3, stride=1, bias=True),
            nn.BatchNorm2d(self.num_filters),   # after (1,18)
            nn.ReLU(inplace=True)
        )

        self.linear1 = nn.Linear(18, self.features_per_filter)  # (18,4)
        self.linear2 = nn.Linear(18, self.features_per_filter)
        self.linear3 = nn.Linear(18, self.features_per_filter)
        self.linear4 = nn.Linear(18, self.features_per_filter)
        self.linear5 = nn.Linear(18, self.features_per_filter)
        self.linear6 = nn.Linear(18, self.features_per_filter)
        self.linear7 = nn.Linear(18, self.features_per_filter)
        self.linear8 = nn.Linear(18, self.features_per_filter)

        self.linear9 = nn.Linear(18, self.features_per_filter)
        self.linear10 = nn.Linear(18, self.features_per_filter)
        self.linear11 = nn.Linear(18, self.features_per_filter)
        self.linear12 = nn.Linear(18, self.features_per_filter)
        self.linear13 = nn.Linear(18, self.features_per_filter)
        self.linear14 = nn.Linear(18, self.features_per_filter)
        self.linear15 = nn.Linear(18, self.features_per_filter)
        self.linear16 = nn.Linear(18, self.features_per_filter)

        self.linear17 = nn.Linear(18, self.features_per_filter)
        self.linear18 = nn.Linear(18, self.features_per_filter)
        self.linear19 = nn.Linear(18, self.features_per_filter)
        self.linear20 = nn.Linear(18, self.features_per_filter)
        self.linear21 = nn.Linear(18, self.features_per_filter)
        self.linear22 = nn.Linear(18, self.features_per_filter)
        self.linear23 = nn.Linear(18, self.features_per_filter)
        self.linear24 = nn.Linear(18, self.features_per_filter)
        self.linear25 = nn.Linear(18, self.features_per_filter)
        self.linear26 = nn.Linear(18, self.features_per_filter)
        self.linear27 = nn.Linear(18, self.features_per_filter)
        self.linear28 = nn.Linear(18, self.features_per_filter)
        self.linear29 = nn.Linear(18, self.features_per_filter)
        self.linear30 = nn.Linear(18, self.features_per_filter)
        self.linear31 = nn.Linear(18, self.features_per_filter)
        self.linear32 = nn.Linear(18, self.features_per_filter)

        self.linear = [self.linear1, self.linear2, self.linear3,
                       self.linear4, self.linear5, self.linear6,
                       self.linear7, self.linear8, self.linear9,
                       self.linear10, self.linear11, self.linear12,
                       self.linear13, self.linear14, self.linear15,
                       self.linear16, self.linear17, self.linear18,
                       self.linear19, self.linear20, self.linear21,
                       self.linear22, self.linear23, self.linear24,
                       self.linear25, self.linear26, self.linear27,
                       self.linear28, self.linear29, self.linear30,
                       self.linear31, self.linear32]

        self.img_channel = img_dim[0]   # the channels of the input image
        self.img_height = img_dim[1]    # the height of the input image
        self.img_width = img_dim[2]     # the width of the input image
        # for reversing the CNN output into (batch,time,channel,height,width)
        self.time_sequence = time_sequence
        # 32 * 4 =128
        self.total_features = self.num_filters * self.features_per_filter

    def forward(self, x):
        """Define forward process of Nvidia CNN_head."""
        # x has the shape (batch size, time Sequence, channel, height, width)
        # flatten the time_sequence*batch_size

        # necessary because the last batch's size is not equal
        # to set batch size
        batch_size = x.shape[0]

        # flatten x (batch_size * time_Sequence)
        x = x.view(-1, self.img_channel, self.img_height, self.img_width)

        # do pic whitening  (pic-mean)/std
        x = _image_standardization(x)

        x = x.view(-1, self.img_channel, self.img_height, self.img_width)
        x = self.conv(x)  # shape (sample_numbers, channels, height, width)

        # after get the result from the conv layer,
        # split the output of each channel, and feed the channels input into
        # the linear layer individually
        # slice one by one , (sample_numbers,1,height, width)
        self.filter_output = list(torch.split(x, 1, dim=1))

        feature_layer_list = []
        for i in range(self.num_filters):
            # (sample_numbers, height, width)
            self.filter_output[i] = torch.squeeze(
                self.filter_output[i], dim=1)

            # flatten the output of each filter, 1*18 = 18
            self.filter_output[i] = self.filter_output[i].view(-1, 18)
            # the output of each filter feed into linear layer
            feats = F.relu(self.linear[i](self.filter_output[i]))
            feature_layer_list.append(feats)

        # concat the features from each filter together
        self.feature_layer = torch.cat(feature_layer_list, 1)

        feature_layer = self.feature_layer.view(
            batch_size, self.time_sequence, self.total_features)

        return feature_layer  # (time_Sequence, batch_size, total_features)

    def count_params(self):
        """Return back how many params CNN_head have."""
        return sum(param.numel() for param in self.parameters())


class ConvolutionHead_ResNet(nn.Module):
    """This class defines ResNet CNN head."""

    # use ResNet18 structure, with less channels.
    def __init__(self, img_dim, time_sequence,
                 num_filters=8, features_per_filter=4):
        """Initialize the object."""
        super(ConvolutionHead_ResNet, self).__init__()

        self.feature_layer = None
        self.filter_output = []
        self.linear = []
        self.num_filters = num_filters
        self.features_per_filter = features_per_filter

        self.in_channel = 24
        # layer before Residual Block  input image (66,200)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),  # (64,198)
            nn.ReLU(inplace=True)
        )
        self.layer1 = self.make_layer(ResBlock, 36, 2, stride=2)  # (32,99)
        self.layer2 = self.make_layer(ResBlock, 48, 2, stride=2)  # (16,50)
        self.layer3 = self.make_layer(ResBlock, 64, 2, stride=2)  # (8,25)
        self.layer4 = self.make_layer(ResBlock, 64, 2, stride=2)  # (4,13)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,
                      self.num_filters,
                      kernel_size=3,
                      stride=1,
                      bias=False),   # (2,11)
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(inplace=True)
        )   # (3,12)

        self.linear1 = nn.Linear(22, self.features_per_filter)  # (22,4)
        self.linear2 = nn.Linear(22, self.features_per_filter)
        self.linear3 = nn.Linear(22, self.features_per_filter)
        self.linear4 = nn.Linear(22, self.features_per_filter)
        self.linear5 = nn.Linear(22, self.features_per_filter)
        self.linear6 = nn.Linear(22, self.features_per_filter)
        self.linear7 = nn.Linear(22, self.features_per_filter)
        self.linear8 = nn.Linear(22, self.features_per_filter)

        self.linear9 = nn.Linear(22, self.features_per_filter)
        self.linear10 = nn.Linear(22, self.features_per_filter)
        self.linear11 = nn.Linear(22, self.features_per_filter)
        self.linear12 = nn.Linear(22, self.features_per_filter)
        self.linear13 = nn.Linear(22, self.features_per_filter)
        self.linear14 = nn.Linear(22, self.features_per_filter)
        self.linear15 = nn.Linear(22, self.features_per_filter)
        self.linear16 = nn.Linear(22, self.features_per_filter)

        self.linear17 = nn.Linear(22, self.features_per_filter)
        self.linear18 = nn.Linear(22, self.features_per_filter)
        self.linear19 = nn.Linear(22, self.features_per_filter)
        self.linear20 = nn.Linear(22, self.features_per_filter)
        self.linear21 = nn.Linear(22, self.features_per_filter)
        self.linear22 = nn.Linear(22, self.features_per_filter)
        self.linear23 = nn.Linear(22, self.features_per_filter)
        self.linear24 = nn.Linear(22, self.features_per_filter)
        self.linear25 = nn.Linear(22, self.features_per_filter)
        self.linear26 = nn.Linear(22, self.features_per_filter)
        self.linear27 = nn.Linear(22, self.features_per_filter)
        self.linear28 = nn.Linear(22, self.features_per_filter)
        self.linear29 = nn.Linear(22, self.features_per_filter)
        self.linear30 = nn.Linear(22, self.features_per_filter)
        self.linear31 = nn.Linear(22, self.features_per_filter)
        self.linear32 = nn.Linear(22, self.features_per_filter)

        self.linear = [self.linear1, self.linear2, self.linear3,
                       self.linear4, self.linear5, self.linear6,
                       self.linear7, self.linear8, self.linear9,
                       self.linear10, self.linear11, self.linear12,
                       self.linear13, self.linear14, self.linear15,
                       self.linear16, self.linear17, self.linear18,
                       self.linear19, self.linear20, self.linear21,
                       self.linear22, self.linear23, self.linear24,
                       self.linear25, self.linear26, self.linear27,
                       self.linear28, self.linear29, self.linear30,
                       self.linear31, self.linear32]

        self.img_channel = img_dim[0]  # the channels of the input image
        self.img_height = img_dim[1]  # the height of the input image
        self.img_width = img_dim[2]  # the width of the input image
        # for reversing the CNN output into (batch,time,channel,height,width)
        self.time_sequence = time_sequence
        # 32 * 4 =128
        self.total_features = self.num_filters * self.features_per_filter

    def make_layer(self, block, channels, num_blocks, stride):
        """Make layers of resblock."""
        strides = [stride] + [1]*(num_blocks-1)
        # create a list [stride,1,1,..,1], the number is:num_blocks-1
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels

        return nn.Sequential(*layers)

    def forward(self, x):
        """Define forward process of ResNet CNN head."""
        # x has the shape (batch size, time Sequence, channel, height, width)
        # flatten the time_sequence*batch_size

        # necessary because the last batch's size is not equal
        # to set batch size
        batch_size = x.shape[0]

        # flatten x (batch_size * time_Sequence)
        x = x.view(-1, self.img_channel, self.img_height, self.img_width)

        # do pic whitening  (pic-mean)/std
        x = _image_standardization(x)

        x = x.view(-1, self.img_channel, self.img_height, self.img_width)
        x = self.conv1(x)  # shape (sample_numbers, channels, height, width)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv2(x)

        # after get the result from the conv layer,
        # split the output of each channel,
        # and feed the channels input into
        # the linear layer individually
        # slice one by one , (sample_numbers,1,height, width)
        self.filter_output = list(torch.split(x, 1, dim=1))

        feature_layer_list = []
        for i in range(self.num_filters):
            # print(filter_output[i].shape)
            # (sample_numbers, height, width)
            self.filter_output[i] = torch.squeeze(
                self.filter_output[i], dim=1)

            # flatten the output of each filter, 2*11 = 22
            self.filter_output[i] = self.filter_output[i].view(-1, 22)

            # the output of each filter feed into linear layer
            feats = F.relu(self.linear[i](self.filter_output[i]))
            feature_layer_list.append(feats)

        # concat the features from each filter together
        self.feature_layer = torch.cat(feature_layer_list, 1)
        feature_layer = self.feature_layer.view(
            batch_size, self.time_sequence, self.total_features)

        return feature_layer  # (time_Sequence, batch_size, total_features)

    def count_params(self):
        """Return how many params CNN_head have."""
        return sum(param.numel() for param in self.parameters())


class ConvolutionHead_AlexNet(nn.Module):
    """This class defines AlexNet CNN head."""

    def __init__(self, img_dim, time_sequence,
                 num_filters=8, features_per_filter=4):
        """Initialize the object."""
        super(ConvolutionHead_AlexNet, self).__init__()
        self.feature_layer = None
        self.filter_output = []
        self.linear = []
        self.num_filters = num_filters
        self.features_per_filter = features_per_filter
        self.conv = nn.Sequential(  # (66, 200)
            nn.Conv2d(3, 24, kernel_size=5, stride=1, padding=2, bias=True),
            # after (66,200)
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (33,100)

            nn.Conv2d(24, 36, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (16，50）

            nn.Conv2d(36, 48, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # （8,25）

            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (4,12)

            nn.Conv2d(64, self.num_filters,
                      kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (2,6)
        )

        self.linear1 = nn.Linear(12, self.features_per_filter)  # (12,4)
        self.linear2 = nn.Linear(12, self.features_per_filter)
        self.linear3 = nn.Linear(12, self.features_per_filter)
        self.linear4 = nn.Linear(12, self.features_per_filter)
        self.linear5 = nn.Linear(12, self.features_per_filter)
        self.linear6 = nn.Linear(12, self.features_per_filter)
        self.linear7 = nn.Linear(12, self.features_per_filter)
        self.linear8 = nn.Linear(12, self.features_per_filter)

        self.linear9 = nn.Linear(12, self.features_per_filter)
        self.linear10 = nn.Linear(12, self.features_per_filter)
        self.linear11 = nn.Linear(12, self.features_per_filter)
        self.linear12 = nn.Linear(12, self.features_per_filter)
        self.linear13 = nn.Linear(12, self.features_per_filter)
        self.linear14 = nn.Linear(12, self.features_per_filter)
        self.linear15 = nn.Linear(12, self.features_per_filter)
        self.linear16 = nn.Linear(12, self.features_per_filter)

        self.linear17 = nn.Linear(12, self.features_per_filter)
        self.linear18 = nn.Linear(12, self.features_per_filter)
        self.linear19 = nn.Linear(12, self.features_per_filter)
        self.linear20 = nn.Linear(12, self.features_per_filter)
        self.linear21 = nn.Linear(12, self.features_per_filter)
        self.linear22 = nn.Linear(12, self.features_per_filter)
        self.linear23 = nn.Linear(12, self.features_per_filter)
        self.linear24 = nn.Linear(12, self.features_per_filter)
        self.linear25 = nn.Linear(12, self.features_per_filter)
        self.linear26 = nn.Linear(12, self.features_per_filter)
        self.linear27 = nn.Linear(12, self.features_per_filter)
        self.linear28 = nn.Linear(12, self.features_per_filter)
        self.linear29 = nn.Linear(12, self.features_per_filter)
        self.linear30 = nn.Linear(12, self.features_per_filter)
        self.linear31 = nn.Linear(12, self.features_per_filter)
        self.linear32 = nn.Linear(12, self.features_per_filter)

        self.linear = [self.linear1, self.linear2, self.linear3,
                       self.linear4, self.linear5, self.linear6,
                       self.linear7, self.linear8, self.linear9,
                       self.linear10, self.linear11, self.linear12,
                       self.linear13, self.linear14, self.linear15,
                       self.linear16, self.linear17, self.linear18,
                       self.linear19, self.linear20, self.linear21,
                       self.linear22, self.linear23, self.linear24,
                       self.linear25, self.linear26, self.linear27,
                       self.linear28, self.linear29, self.linear30,
                       self.linear31, self.linear32]

        self.img_channel = img_dim[0]  # the channels of the input image
        self.img_height = img_dim[1]  # the height of the input image
        self.img_width = img_dim[2]  # the width of the input image
        # for reversing the CNN output into (batch,time,channel,height,width)
        self.time_sequence = time_sequence
        # 32 * 4 =128
        self.total_features = self.num_filters * self.features_per_filter

    def forward(self, x):
        """Define forward process of AlexNet CNN head."""
        # x has the shape (batch size, time Sequence, channel, height, width)
        # flatten the time_sequence*batch_size
        # necessary because the last batch's size
        # is not equal to set batch size
        batch_size = x.shape[0]

        # flatten x (batch_size * time_Sequence)
        x = x.view(-1, self.img_channel, self.img_height, self.img_width)

        # do pic whitening  (pic-mean)/std
        x = _image_standardization(x)

        x = x.view(-1, self.img_channel, self.img_height, self.img_width)
        x = self.conv(x)  # shape (sample_numbers, channels, height, width)

        # after get the result from the conv layer,
        # split the output of each channel,
        # and feed the channels input into the linear layer individually
        # slice one by one , (sample_numbers,1,height, width)
        self.filter_output = list(torch.split(x, 1, dim=1))

        feature_layer_list = []
        for i in range(self.num_filters):
            # print(filter_output[i].shape)
            self.filter_output[i] = torch.squeeze(
                self.filter_output[i], dim=1)
            # (sample_numbers, height, width)

            # flatten the output of each filter, 12
            self.filter_output[i] = self.filter_output[i].view(-1, 12)

            # the output of each filter feed into linear layer
            feats = F.relu(self.linear[i](self.filter_output[i]))
            feature_layer_list.append(feats)

        # concat the features from each filter together
        self.feature_layer = torch.cat(feature_layer_list, 1)

        feature_layer = self.feature_layer.view(
            batch_size, self.time_sequence, self.total_features)

        return feature_layer  # (time_Sequence, batch_size, total_features)

    def count_params(self):
        """Return back how many params CNN_head have."""
        return sum(param.numel() for param in self.parameters())


if __name__ == "__main__":
    s = (3, 128, 256)
    a = ConvolutionHead_Nvidia(
        s,
        1,
        num_filters=32,
        features_per_filter=4)
    b = ConvolutionHead_ResNet(
        s,
        16,
        num_filters=32,
        features_per_filter=4)
    c = ConvolutionHead_AlexNet(
        s,
        16,
        num_filters=32,
        features_per_filter=4)
    print(a.count_params())
    print(b.count_params())
    print(c.count_params())
