import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.dim_z = params.z_size
        self.ngpu = params.ngpu
        self.lin = nn.Parameter(0.1 * torch.randn(self.dim_z))
        self.lin = self.lin/torch.norm(self.lin)

    def forward(self, z, alpha):
        d = torch.mul((self.lin/torch.norm(self.lin)).reshape(1,-1).transpose(0,1), alpha).transpose(0,1)
        return z + d
