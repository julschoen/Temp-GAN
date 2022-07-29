import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.p = params
        self.dim_z = params.z_size
        self.ngpu = params.ngpu

        self.linear = nn.Linear(1, self.dim_z, bias=True)
        self.linear.weight.data = torch.randn_like(self.linear.weight.data)
        
        if self.p.norm:
            self.linear.weight.data = self.linear.weight.data/torch.norm(self.linear.weight.data)
        else:
            self.linear.weight.data = 0.1 * self.linear.weight.data

        self.lin = torch.randn(self.dim_z)
        if self.p.norm:
            self.lin = self.lin/torch.norm(self.lin)
        self.lin = nn.Parameter(self.lin)

    def forward(self, z, alpha):
        alpha.view(-1,1)
        if self.p.norm:
            input_norm = torch.norm(alpha, dim=1, keepdim=True)
            out = self.linear(alpha)
            d = (input_norm / torch.norm(out, dim=1, keepdim=True)) * out
            print(d.shape)
            print(torch.norm(d))
            #d = torch.mul((self.lin/torch.norm(self.lin)).reshape(1,-1).transpose(0,1), alpha).transpose(0,1)
        else:
            d  = self.linear(input)
            print(d.shape)
            #d = torch.mul(self.lin.reshape(1,-1).transpose(0,1), alpha).transpose(0,1)
        return z + d
