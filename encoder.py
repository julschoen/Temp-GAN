import torch 
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
import functools
from utils import Attention, DBlock, snconv3d, snlinear

class Encoder(nn.Module):
  def __init__(self, params):
    super(Encoder, self).__init__()
    self.p = params
    self.main = nn.Sequential(
        # input is 128 x 128 x 128
        SpectralNorm(nn.Conv3d(nc, ndf, 4, stride=2, padding=1, bias=False)), 
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf) x 64 x 64 x 64
        SpectralNorm(nn.Conv3d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False)),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*2) x 32 x 32 x 32
        SpectralNorm(nn.Conv3d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False)),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*4) x 16 x 16 x 16
        SpectralNorm(nn.Conv3d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False)),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*8) x 8 x 8 x 8
        SpectralNorm(nn.Conv3d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False)),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ndf*16) x 4 x 4 x 4
        SpectralNorm(nn.Conv3d(ndf * 16, self.p.z_size, (4,4,4), stride=1, padding=0, bias=False)),
        nn.Tanh()
    )
    self.init_weights()

  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv3d)
          or isinstance(module, nn.Linear)):
        init.orthogonal_(module.weight)
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for D''s initialized parameters: %d' % self.param_count)

  def forward(self, x):
    return self.main(x)