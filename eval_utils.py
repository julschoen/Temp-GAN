import numpy as np
from scipy.linalg import sqrtm
from collections import OrderedDict
import torch
import pytorch_fid_wrapper as FID
from pytorch_msssim import MS_SSIM
from torch.cuda.amp import autocast

def psnr(real, fake):
    with torch.no_grad():
        with autocast():
            real, fake = real+1, fake+1 
            mse = torch.mean(torch.square((real - fake)))
            if(mse == 0):
                return 100
            psnr_ = 10 * (torch.log(4/mse)/torch.log(torch.Tensor([10]))).item()
    return psnr_

def ssim(real, fake):
    with torch.no_grad():
        real = (real+1)/2
        fake = (fake+1)/2
        ms_ssim_module = MS_SSIM(data_range=1, win_size=3, size_average=True, channel=1, spatial_dims=3)
        ms_ssim_ = ms_ssim_module(real.cpu().to(torch.float32), fake.cpu().to(torch.float32)).item()
    return ms_ssim_

def fid(real, fake, device):
    FID.set_config(device=device)
    real.to(device)
    fake.to(device)
    with torch.no_grad():
        with autocast():
            fid_ax = FID.fid(
                    torch.reshape(fake.to(torch.float32), (-1,1,128,128)).expand(-1,3,-1,-1), 
                    real_images=torch.reshape(real.to(torch.float32), (-1,1,128,128)).expand(-1,3,-1,-1)
                    )

            fid_cor = FID.fid(
                    torch.reshape(fake.to(torch.float32).transpose(2,3), (-1,1,64,128)).expand(-1,3,-1,-1), 
                    real_images=torch.reshape(real.to(torch.float32).transpose(2,3), (-1,1,64,128)).expand(-1,3,-1,-1)
                    )
            fid_sag = FID.fid(
                    torch.reshape(fake.to(torch.float32).transpose(4,2), (-1,1,128,64)).expand(-1,3,-1,-1), 
                    real_images=torch.reshape(real.to(torch.float32).transpose(4,2), (-1,1,128,64)).expand(-1,3,-1,-1)
                    )
    return fid_ax, fid_cor, fid_sag
