from functools import partial

import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import conv3x3x3, downsample_basic_block, BasicBlock, Bottleneck, Conv3_1d


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        #self.conv1 = Conv3_1d(1, 64, kernel_size=(3,7,7,7), stride=(1, 2, 2, 2), padding=(0, 3, 3, 3), bias=False)
        self.extractor = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1),
            self._make_layer(block, 64, layers[0]),
            self._make_layer(block, 128, layers[1], stride=2),
            self._make_layer(block, 256, layers[2], stride=1, dilation=2),
            self._make_layer(BasicBlock, 512, layers[3], stride=1, dilation=4)
        ) 
        
        self.head = nn.Linear(512*3, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, expansion=None, stride=1, dilation=1):
        if expansion is None:
            expansion = block.expansion
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm3d(planes * expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        h1 = torch.sum(self.extractor(x[:,0].unsqueeze(1)), [2, 3, 4])
        h2 = torch.sum(self.extractor(x[:,1].unsqueeze(1)), [2, 3, 4])
        h3 = torch.sum(self.extractor(x[:,2].unsqueeze(1)), [2, 3, 4])
        h = torch.concat((h1,h2,h3), dim=1) 
        return self.head(h)



def Discriminator(params, **kwargs):
    #### ResNet-10
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    #### ResNet-18
    #model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    #### ResNet-50
    #model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    #### ResNet-152
    #model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
