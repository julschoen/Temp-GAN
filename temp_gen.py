import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.dim_z = params.z_size
        self.ngpu = params.ngpu
        self.lin = nn.Linear(in_features=self.dim_z, out_features=self.dim_z)
        self.lin.weight.data = 0.01 * torch.randn_like(self.lin.weight.data)

    def forward(self, input):
        return self.lin(input)
