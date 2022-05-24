import numpy as np
from scipy.linalg import sqrtm
from collections import OrderedDict
import torch
from FID_ResNet import resnet50
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
        ms_ssim_module = MS_SSIM(data_range=1, win_size=7, size_average=True, channel=1, spatial_dims=3)
        ms_ssim_ = ms_ssim_module(real.cpu().to(torch.float32), fake.cpu().to(torch.float32)).item()
    return ms_ssim_
 
def fid_3d(model, real, fake):
    # calculate activations
    with torch.no_grad():
        with autocast():
            act1 = model(real.cuda()).mean(dim=(2,3,4)).detach().cpu().numpy()
            act2 = model(fake.cuda()).mean(dim=(2,3,4)).detach().cpu().numpy() 
            # calculate mean and covariance statistics
            mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
            mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
            # calculate sum squared difference between means
            ssdiff = np.sum((mu1 - mu2)**2.0)
            # calculate sqrt of product between cov
            covmean = sqrtm(sigma1.dot(sigma2))
            # check and correct imaginary numbers from sqrt
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            # calculate score
            fid_ = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid_

def get_fid_model(path):
    fid_model = resnet50()
    state = torch.load(path)['state_dict']

    new_state_dict = OrderedDict()
    for k, v in state.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    fid_model.load_state_dict(new_state_dict)
    return fid_model

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
                    torch.reshape(fake.to(torch.float32).transpose(2,3), (-1,1,128,128)).expand(-1,3,-1,-1), 
                    real_images=torch.reshape(real.to(torch.float32).transpose(2,3), (-1,1,128,128)).expand(-1,3,-1,-1)
                    )
            fid_sag = FID.fid(
                    torch.reshape(fake.to(torch.float32).transpose(4,2), (-1,1,128,128)).expand(-1,3,-1,-1), 
                    real_images=torch.reshape(real.to(torch.float32).transpose(4,2), (-1,1,128,128)).expand(-1,3,-1,-1)
                    )
    return fid_ax, fid_cor, fid_sag
