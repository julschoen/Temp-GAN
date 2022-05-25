import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.dim_z = params.z_size
        self.ngpu = params.ngpu
        self.lin = nn.Linear(in_features=512, out_features=1)
        self.pad = nn.ConstantPad1d((0,self.dim_z-1), 0)

    def forward(self, input):
        output = self.lin(input)
        output = input + self.pad(output)
        return output
