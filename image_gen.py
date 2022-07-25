import torch 
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from utils import Attention, GBlock, snconv3d, snlinear

class Generator(nn.Module):
  def __init__(self, params):
    super(Generator, self).__init__()
    self.p = params
    self.dim_z = self.p.z_size

    self.arch = {'in_channels' :  [item * self.p.filterG for item in [16, 16, 8, 4, 2]],
             'out_channels' : [item * self.p.filterG for item in [16, 8, 4,  2, 1]],
             'upsample' : [True] * 5,
             'resolution' : [8, 16, 32, 64, 128],
             'attention' : {2**i: (2**i in [int(item) for item in '32'.split('_')]) for i in range(3,8)}}

    if self.p.lidc:
      self.linear = snlinear(self.p.z_size, self.arch['in_channels'][0] * 64)
    else:
      self.linear = snlinear(self.p.z_size, self.arch['in_channels'][0] * 32)
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[GBlock(in_channels=self.arch['in_channels'][index],
                             out_channels=self.arch['out_channels'][index],
                             upsample=(functools.partial(F.interpolate, scale_factor=2)
                                       if self.arch['upsample'][index] else None))]]
      if self.arch['attention'][self.arch['resolution'][index]]:
        self.blocks[-1] += [Attention(self.arch['out_channels'][index])]

    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    self.output_layer = nn.Sequential(nn.BatchNorm3d(self.arch['out_channels'][-1]),
                                    nn.ReLU(inplace=True),
                                    snconv3d(self.arch['out_channels'][-1], 1))

    self.init_weights()

  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv3d) 
          or isinstance(module, nn.Linear) 
          or isinstance(module, nn.Embedding)):
        init.orthogonal_(module.weight)
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for G''s initialized parameters: %d' % self.param_count)

  def forward(self, z):
    # First linear layer
    h = self.linear(z)
    # Reshape
    if self.p.lidc:
      h = h.view(h.size(0), -1, 4, 4, 4)
    else:
      h = h.view(h.size(0), -1, 2, 4, 4)
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    return torch.tanh(self.output_layer(h))
