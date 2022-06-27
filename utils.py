import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
import functools
from torch.nn import Parameter as P

class Conv3_1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3,3), stride=(1, 1, 1, 1),
                        padding=(0, 1, 1, 1), dilation=(1, 1, 1, 1), bias=True):
        super().__init__()
        t = kernel_size[0]
        d = (kernel_size[1] + kernel_size[2] + kernel_size[3])//2
        self.in_channels = in_channels
        self.out_channels = out_channels

        #Hidden size estimation to get a number of parameter similar to the 3d case
        self.hidden_size = int((t*d**2*in_channels*out_channels)/(d**2*in_channels+t*out_channels))

        self.conv3d = nn.Conv3d(in_channels, self.hidden_size, kernel_size[1:], stride[1:], padding[1:], bias=bias)
        self.conv1d = nn.Conv1d(self.hidden_size, out_channels, kernel_size[0], stride[0], padding[0], bias=bias)

    def forward(self, x):
        #3D convolution
        b, t, c, d1, d2, d3 = x.size()
        x = x.view(b*t, c, d1, d2, d3)
        x = F.relu(self.conv3d(x))
        
        #1D convolution
        c, dr1, dr2, dr3 = x.size(1), x.size(2), x.size(3), x.size(4)
        x = x.view(b, t, c, dr1, dr2, dr3)
        x = x.permute(0, 3, 4, 5, 2, 1).contiguous()
        x = x.view(b*dr1*dr2*dr3, c, t)
        x = self.conv1d(x)

        #Final output
        out_c, out_t = x.size(1), x.size(2)
        x = x.view(b, dr1, dr2, dr3, out_c, out_t)
        x = x.permute(0, 4, 5, 1, 2, 3).contiguous()
        return x.squeeze()


################# BigGAN ######################
def snconv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    return SpectralNorm(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, bias=bias))

def snlinear(in_features, out_features):
    return SpectralNorm(nn.Linear(in_features=in_features, out_features=out_features))

class Attention(nn.Module):
  def __init__(self, ch):
    super(Attention, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.theta = snconv3d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.phi = snconv3d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
    self.g = snconv3d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
    self.o = snconv3d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)
  def forward(self, x, y=None):
    # Apply convs
    theta = self.theta(x)
    phi = F.max_pool3d(self.phi(x), [2,2,2], stride=2)
    g = F.max_pool3d(self.g(x), [2,2,2], stride=2)    
    # Perform reshapes
    theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3] * x.shape[4])
    phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] * x.shape[4] // 8)
    g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] * x.shape[4] // 8)
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.permute(0,2,1), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.permute(0,2,1)).view(-1, self.ch // 2, x.shape[2], x.shape[3], x.shape[4]))
    return self.gamma * o + x

class GBlock(nn.Module):
  def __init__(self, in_channels, out_channels, upsample=None, channel_ratio=4):
    super(GBlock, self).__init__()
    
    self.in_channels, self.out_channels = in_channels, out_channels
    self.hidden_channels = self.in_channels // channel_ratio
    
    # Conv layers
    self.conv1 = snconv3d(self.in_channels, self.hidden_channels, 
                                 kernel_size=1, padding=0)
    self.conv2 = snconv3d(self.hidden_channels, self.hidden_channels)
    self.conv3 = snconv3d(self.hidden_channels, self.hidden_channels)
    self.conv4 = snconv3d(self.hidden_channels, self.out_channels, 
                                 kernel_size=1, padding=0)
    # Batchnorm layers
    self.bn1 = nn.BatchNorm3d(self.in_channels)
    self.bn2 = nn.BatchNorm3d(self.hidden_channels)
    self.bn3 = nn.BatchNorm3d(self.hidden_channels)
    self.bn4 = nn.BatchNorm3d(self.hidden_channels)
    # upsample layers
    self.upsample = upsample
    self.activation = nn.ReLU(inplace=True)

  def forward(self, x):
    # Project down to channel ratio
    h = self.conv1(self.activation(self.bn1(x)))
    # Apply next BN-ReLU
    h = self.activation(self.bn2(h))
    if self.in_channels != self.out_channels:
      x = x[:, :self.out_channels]   
    # Upsample both h and x at this point
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    # 3x3 convs
    h = self.conv2(h)
    h = self.conv3(self.activation(self.bn3(h)))
    # Final 1x1 conv
    h = self.conv4(self.activation(self.bn4(h)))
    return h + x

class DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, wide=True, preactivation=True,
               downsample=None, channel_ratio=4):
    super(DBlock, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
    self.hidden_channels = self.out_channels // channel_ratio

    self.preactivation = preactivation
    self.activation = nn.ReLU(inplace=True)
    self.downsample = downsample
        
    # Conv layers
    self.conv1 = snconv3d(self.in_channels, self.hidden_channels, 
                                 kernel_size=1, padding=0)
    self.conv2 = snconv3d(self.hidden_channels, self.hidden_channels)
    self.conv3 = snconv3d(self.hidden_channels, self.hidden_channels)
    self.conv4 = snconv3d(self.hidden_channels, self.out_channels, 
                                 kernel_size=1, padding=0)
                                 
    self.learnable_sc = True if (in_channels != out_channels) else False
    if self.learnable_sc:
      self.conv_sc = snconv3d(in_channels, out_channels - in_channels, 
                                     kernel_size=1, padding=0)
  def shortcut(self, x):
    if self.downsample:
      x = self.downsample(x)
    if self.learnable_sc:
      x = torch.cat([x, self.conv_sc(x)], 1)    
    return x
    
  def forward(self, x):
    # 1x1 bottleneck conv
    h = self.conv1(F.relu(x))
    # 3x3 convs
    h = self.conv2(self.activation(h))
    h = self.conv3(self.activation(h))
    # relu before downsample
    h = self.activation(h)
    # downsample
    if self.downsample:
      h = self.downsample(h)     
    # final 1x1 conv
    h = self.conv4(h)
    return h + self.shortcut(x)


################### 3D-ResNet #######################
def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
