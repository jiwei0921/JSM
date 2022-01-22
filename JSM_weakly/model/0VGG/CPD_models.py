import torch
import torch.nn as nn

from model.vgg import B2_VGG


class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_x1 = nn.Conv2d(channel*8, channel, 3, padding=1)
        self.conv_x2 = nn.Conv2d(channel*8, channel, 3, padding=1)
        self.conv_x3 = nn.Conv2d(channel*4, channel, 3, padding=1)

        self.conv_upsample1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = nn.Conv2d(channel, channel, 3, padding=1)

        self.conv_concat2 = nn.Conv2d(2*channel, channel, 3, padding=1)
        self.conv_concat3 = nn.Conv2d(2*channel, channel, 3, padding=1)
        self.conv4 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv5 = nn.Conv2d(channel, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x1, x2, x3):
        # x1: 1/16 x2: 1/8 x3: 1/4
        x1_1 = self.conv_x1(x1)
        x2_1 = self.conv_x2(x2)
        x3_1 = self.conv_x3(x3)

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        fea = x
        x = self.conv5(x)

        return x, fea


class CPD_VGG(nn.Module):
    def __init__(self, channel=64):
        super(CPD_VGG, self).__init__()
        self.vgg = B2_VGG()
        self.agg1 = aggregation(channel)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        x1 = self.vgg.conv1(x)
        x2 = self.vgg.conv2(x1)
        x3 = self.vgg.conv3(x2)

        x3_1 = x3
        x4_1 = self.vgg.conv4_1(x3_1)
        x5_1 = self.vgg.conv5_1(x4_1)

        detection, feature = self.agg1(x5_1, x4_1, x3_1)

        return self.upsample(detection), self.upsample(feature)