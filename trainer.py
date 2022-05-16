import os
import numpy as np
import pytorch_fid_wrapper as FID

import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast, GradScaler

import torchvision
import torchvision.utils as vutils

from image_disc import Discriminator as ImD
from temp_disc import Discriminator as TempD
from image_gen import Generator as ImG
from temp_gen import Generator as TempG


class Trainer(object):
    def __init__(self, dataset, params):
        ### Misc ###
        self.p = params
        self.device = params.device

        ### Make Dirs ###
        self.log_dir = params.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.models_dir = os.path.join(self.log_dir, 'models')
        self.images_dir = os.path.join(self.log_dir, 'images')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        ### Make Models ###
        self.imD = ImD(self.p).to(self.device)
        self.tempD = TempD(self.p).to(self.device)
        self.imG = ImG(self.p).to(self.device)
        self.tempG = TempG(self.p).to(self.device)
        if self.p.ngpu>1:
            self.imD = nn.DataParallel(self.imD)
            self.tempD = nn.DataParallel(self.tempD)
            self.imG = nn.DataParallel(self.imG)
            self.tempG = nn.DataParallel(self.tempG)

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.p.lrD,
                                         betas=(0., 0.9))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.p.lrG,
                                         betas=(0., 0.9))

        self.scalerD = GradScaler()
        self.scalerG = GradScaler()

        ### Make Data Generator ###
        self.generator_train = DataLoader(dataset, batch_size=self.p.batch_size, shuffle=True, num_workers=4, drop_last=True)

        ### Prep Training
        self.fixed_test_noise = None
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.fid = []
        self.fid_epoch = []

    def inf_train_gen(self):
        while True:
            for data in self.generator_train:
                yield data
        
    def log_train(self, step, fake, real, D_x, D_G_z1, D_G_z2):
        with torch.no_grad():
            self.fid.append(
                FID.fid(
                    torch.reshape(fake.to(torch.float32), (-1,1,128,128)).expand(-1,3,-1,-1), 
                    real_images=torch.reshape(real.to(torch.float32), (-1,1,128,128)).expand(-1,3,-1,-1)
                    )
                )

        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tFID %.4f'
                    % (step, self.p.niters,
                        self.D_losses[-1], self.G_losses[-1], D_x, D_G_z1, D_G_z2, self.fid[-1]))

    def log_interpolation(self, step):
        noise = torch.randn(self.p.batch_size, self.p.z_size, 1, 1,1,
                            dtype=torch.float, device=self.device)
        if self.fixed_test_noise is None:
            self.fixed_test_noise = noise.clone()
    
        with torch.no_grad():
            fake = self.netG(self.fixed_test_noise).detach().cpu()
        torchvision.utils.save_image(
            vutils.make_grid(torch.reshape(fake, (-1,1,128,128)), padding=2, normalize=True)
            , os.path.join(self.images_dir, f'{step}.png'))

    def start_from_checkpoint(self):
        step = 0
        checkpoint = os.path.join(self.models_dir, 'checkpoint.pt')
        if os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint)
            step = state_dict['step']
            self.netG.load_state_dict(state_dict['modelG_state_dict'])
            self.netD.load_state_dict(state_dict['modelD_state_dict'])

            self.optimizerG.load_state_dict(state_dict['optimizerG_state_dict'])
            self.optimizerD.load_state_dict(state_dict['optimizerD_state_dict'])

            self.G_losses = state_dict['lossG']
            self.D_losses = state_dict['lossD']
            self.fid_epoch = state_dict['fid']
            print('starting from step {}'.format(step))
        return step

    def save_checkpoint(self, step):
        torch.save({
        'step': step,
        'modelG_state_dict': self.netG.state_dict(),
        'modelD_state_dict': self.netD.state_dict(),
        'optimizerG_state_dict': self.optimizerG.state_dict(),
        'optimizerD_state_dict': self.optimizerD.state_dict(),
        'lossG': self.G_losses,
        'lossD': self.D_losses,
        'fid': self.fid_epoch,
        }, os.path.join(self.models_dir, 'checkpoint.pt'))

    def log(self, step, fake, real, D_x, D_G_z1, D_G_z2):
        if step % self.p.steps_per_log == 0:
            self.log_train(step, fake, real, D_x, D_G_z1, D_G_z2)

        if step % self.p.steps_per_img_log == 0:
            self.log_interpolation(step)

    def log_final(self, step, fake, real, D_x, D_G_z1, D_G_z2):
        self.log_train(step, fake, real, D_x, D_G_z1, D_G_z2)
        self.log_interpolation(step)
        self.save_checkpoint(step)

    def step_d

    def train(self):
        step_done = self.start_from_checkpoint()
        FID.set_config(device=self.device)
        one = torch.FloatTensor([1]).to(self.device)
        mone = one * -1
        gen = self.inf_train_gen()

        print("Starting Training...")
        for i in range(step_done, self.p.niters):
            for p in self.imD.parameters():
                p.requires_grad = True
            for _ in range(self.p.iterD):    
                data = next(gen)
                real = data.to(self.device).unsqueeze(dim=1)
                self.imD.zero_grad()

                with autocast():
                    noise = torch.randn(real.shape[0], self.p.z_size, 1, 1,1,
                                dtype=torch.float, device=self.device)
                    fake = self.netG(noise)
                    errD_real = (nn.ReLU()(1.0 - self.netD(real))).mean()
                    errD_fake = (nn.ReLU()(1.0 + self.netD(fake))).mean()
                    errD = errD_fake + errD_real
                self.scalerD.scale(errD).backward()
                self.scalerD.step(self.optimizerD)
                self.scalerD.update()

            for p in self.netD.parameters():
                p.requires_grad = False

            self.netG.zero_grad()
            
            with autocast():
                noise = torch.randn(real.shape[0], self.p.z_size, 1, 1,1,
                            dtype=torch.float, device=self.device)
                fake = self.netG(noise)
                errG = -self.netD(fake).mean()

            self.scalerG.scale(errG).backward()
            self.scalerG.step(self.optimizerG)
            self.scalerG.update()

            self.G_losses.append(errG.item())
            self.D_losses.append(errD.item())

            self.log(i, fake, real, errD_real.item(), errD_fake.item(), errG.item())
            if i%100 == 0 and i>0:
                self.fid_epoch.append(np.array(self.fid).mean())
                self.fid = []
                self.save_checkpoint(i)
        
        self.log_final(i, fake, real, errD_real.item(), errD_fake.item(), errG.item())
        print('...Done')
