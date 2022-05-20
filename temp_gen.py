import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        nz = params.z_size
        print(nz)
        self.ngpu = params.ngpu
        self.lin = nn.Linear(in_features=nz, out_features=nz)

    def forward(self, input):
        print(input.shape)
        output = self.lin(input)
        return output
