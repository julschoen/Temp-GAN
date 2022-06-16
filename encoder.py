import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils.spectral_norm as SpectralNorm
from utils import Attention as Self_Attn


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        conv_dim = params.filterD
        Normalization = SpectralNorm
        layer1 = []
        layer2 = []
        layer3 = []

        layer1.append(Normalization(nn.Conv3d(1, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))
        curr_dim = conv_dim

        layer2.append(Normalization(nn.Conv3d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(Normalization(nn.Conv3d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        
        layer4 = []
        layer4.append(Normalization(nn.Conv3d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        self.l4 = nn.Sequential(*layer4)
        self.attn1 = Self_Attn(curr_dim*2)
        curr_dim = curr_dim * 2

        layer4 = []
        layer4.append(Normalization(nn.Conv3d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        self.l5 = nn.Sequential(*layer4)
        curr_dim = curr_dim * 2

        self.last = nn.Linear(curr_dim, params.z_size)


    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.attn1(out)
        out = self.l5(out)
        out = torch.sum(out, [2, 3, 4])
        out = self.last(out)
        return out.squeeze()