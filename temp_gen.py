import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.dim_z = params.z_size
        self.ngpu = params.ngpu
        self.lin = torch.randn(self.dim_z)
        #self.lin = nn.Parameter(self.lin/torch.norm(self.lin))
        self.lin = nn.Parameter(self.lin)

    def forward(self, z, alpha):
        #d = torch.mul((self.lin/torch.norm(self.lin)).reshape(1,-1).transpose(0,1), alpha).transpose(0,1)
        d = torch.mul(self.lin.reshape(1,-1).transpose(0,1), alpha).transpose(0,1)
        return z + d
