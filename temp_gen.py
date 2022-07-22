import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.p = params
        self.dim_z = params.z_size
        self.ngpu = params.ngpu
        self.lin = torch.randn(self.dim_z)
        if self.p.norm:
            self.lin = self.lin/torch.norm(self.lin)
        self.lin = nn.Parameter(self.lin)

    def forward(self, z, alpha):
        if self.p.norm:
            d = torch.mul((self.lin/torch.norm(self.lin)).reshape(1,-1).transpose(0,1), alpha).transpose(0,1)
        else:
            d = torch.mul(self.lin.reshape(1,-1).transpose(0,1), alpha).transpose(0,1)
        return z + d
