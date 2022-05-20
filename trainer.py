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
from one_disc import Discriminator as TempD
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
        #self.imD = ImD(self.p).to(self.device)
        self.tempD = TempD(self.p).to(self.device)
        self.imG = ImG(self.p).to(self.device)
        self.tempG = TempG(self.p).to(self.device)
        if self.p.ngpu>1:
            #self.imD = nn.DataParallel(self.imD)
            self.tempD = nn.DataParallel(self.tempD)
            self.imG = nn.DataParallel(self.imG)
            self.tempG = nn.DataParallel(self.tempG)

        #self.optimizerImD = optim.Adam(self.imD.parameters(), lr=self.p.lrD,
        #                                 betas=(0., 0.9))
        self.optimizerImG = optim.Adam(self.imG.parameters(), lr=self.p.lrG,
                                         betas=(0., 0.9))

        self.optimizerTempD = optim.Adam(self.tempD.parameters(), lr=self.p.lrD,
                                         betas=(0., 0.9))
        self.optimizerTempG = optim.Adam(self.tempG.parameters(), lr=self.p.lrG,
                                         betas=(0., 0.9))

        #self.scalerImD = GradScaler()
        self.scalerImG = GradScaler()
        self.scalerTempD = GradScaler()
        self.scalerTempG = GradScaler()

        ### Make Data Generator ###
        self.generator_train = DataLoader(dataset, batch_size=self.p.batch_size, shuffle=True, num_workers=4, drop_last=True)

        ### Prep Training
        self.fixed_test_noise = None
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.triplet_losses = []
        self.fid = []
        self.fid_epoch = []
        self.reg_loss = nn.MSELoss()

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
            self.imG.load_state_dict(state_dict['imG'])
            self.imD.load_state_dict(state_dict['imD'])

            self.tempG.load_state_dict(state_dict['tempG'])
            self.tempD.load_state_dict(state_dict['tempD'])

            self.optimizerImG.load_state_dict(state_dict['optimizerImG'])
            self.optimizerImD.load_state_dict(state_dict['optimizerImD'])

            self.optimizerTempG.load_state_dict(state_dict['optimizerTempG'])
            self.optimizerTempD.load_state_dict(state_dict['optimizerTempD'])

            self.G_losses = state_dict['lossG']
            self.D_losses = state_dict['lossD']
            self.triplet_losses = state_dict['triplet']
            self.fid_epoch = state_dict['fid']
            print('starting from step {}'.format(step))
        return step

    def save_checkpoint(self, step):
        torch.save({
        'step': step,
        'imG': self.imG.state_dict(),
        'imD': self.imD.state_dict(),
        'tempG': self.tempG.state_dict(),
        'tempD': self.tempD.state_dict(),
        'optimizerImG': self.optimizerImG.state_dict(),
        'optimizerImD': self.optimizerImD.state_dict(),
        'optimizerTempG': self.optimizerTempG.state_dict(),
        'optimizerTempD': self.optimizerTempD.state_dict(),
        'lossG': self.G_losses,
        'lossD': self.D_losses,
        'triplet': self.triplet_losses,
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

    def sample_g(self, grad):
        if grad:
            for p in self.tempG.parameters():
                p.requires_grad = True
            for p in self.imG.parameters():
                p.requires_grad = True

            self.imG.zero_grad()
            self.tempG.zero_grad()
        ims = None
        zs = None
        inds = None
        with autocast():
            for _ in range(self.p.batch_size):
                z = torch.randn(1, self.p.z_size, dtype=torch.float, device=self.device)
                for i in range(torch.randint(low=2, high=11, size=())):
                    z = torch.concat(
                        (z, self.tempG(z[-1].unsqueeze(0)).reshape(1,-1))
                    )
                ind = torch.randint(len(z), (3,))
                z = z[ind]
                im = self.imG(z)
                if ims is None:
                    ims = im.reshape(1,3,128,128,-1)
                    zs = z[0].reshape(1,-1)
                    inds = ind.reshape(1,3)
                else:
                    ims = torch.concat((ims,im.reshape(1,3,128,128,-1)))
                    zs = torch.concat((zs,z[0].reshape(1,-1)))
                    inds = torch.concat((inds, ind.reshape(1,3)))

        return ims, zs, inds

    def step_imD(self, real, fake, noise):
        for p in self.imD.parameters():
            p.requires_grad = True
        
        self.imD.zero_grad()
        with autocast():
            disc_fake, zs = self.imD(fake)
            disc_real, _ = self.imD(real)
            errD_real = (nn.ReLU()(1.0 - disc_real)).mean()
            errD_fake = (nn.ReLU()(1.0 + disc_fake)).mean()
            rec_loss = self.reg_loss(zs, noise)
            errImD = errD_fake + errD_real + rec_loss
        self.scalerImD.scale(errImD).backward()
        self.scalerImD.step(self.optimizerImD)
        self.scalerImD.update()

        for p in self.imD.parameters():
            p.requires_grad = False

    def step_tempD(self, real, fake):
        for p in self.tempD.parameters():
            p.requires_grad = True
        self.tempD.zero_grad()
        with autocast():
            disc_fake, triplet = self.tempD(fake)
            disc_real, triplet = self.tempD(real)
            errD_real = (nn.ReLU()(1.0 - disc_real)).mean()
            errD_fake = (nn.ReLU()(1.0 + disc_fake)).mean()
            triplet_real = self.reg_loss(zs, noise)
            triplet_fake = self.reg_loss(zs, noise)
            errTempD = errD_fake + errD_real + triplet_real + triplet_fake
        self.scalerTempD.scale(errTempD).backward()
        self.scalerTempD.step(self.optimizerTempD)
        self.scalerTempD.update()

        for p in self.tempD.parameters():
            p.requires_grad = False

    def step_D(self, real, fake, noise, ind):
        for p in self.tempD.parameters():
            p.requires_grad = True
        self.tempD.zero_grad()
        with autocast():
            disc_fake, zs, triplet = self.tempD(fake)
            disc_real, zs, triplet = self.tempD(real)
            errD_real = (nn.ReLU()(1.0 - disc_real)).mean()
            errD_fake = (nn.ReLU()(1.0 + disc_fake)).mean()
            err_rec_z = self.reg_loss(zs, noise)

            #triplet_real = self.reg_loss(zs, noise)
            #triplet_fake = self.reg_loss(zs, noise)

            errTempD = errD_fake + errD_real + err_rec_z#+ triplet_real + triplet_fake
        self.scalerTempD.scale(errTempD).backward()
        self.scalerTempD.step(self.optimizerTempD)
        self.scalerTempD.update()

        for p in self.tempD.parameters():
            p.requires_grad = False

    def step_G(self):
        fake, noise, ind = self.sample_g(grad=True)

        with autocast():
            #disc_im_fake, _ = self.imD(fake)
            disc_temp_fake, zs, triplet = self.tempD(fake)
            #errImG = -disc_fake.mean() - disc_temp_fake.mean()
            errImG = - disc_temp_fake.mean()

            triplet_loss = self.reg_loss(zs, noise)

            errTempG = -triplet_loss.mean() - disc_temp_fake.mean()

        self.scalerImG.scale(errImG).backward(retain_graph=True)
        self.scalerImG.step(self.optimizerImG)
        self.scalerImG.update()

        self.scalerTempG.scale(errTempG).backward()
        self.scalerTempG.step(self.optimizerTempG)
        self.scalerTempG.update()

        for p in self.tempG.parameters():
                p.requires_grad = False
        for p in self.imG.parameters():
            p.requires_grad = False

    def train(self):
        step_done = self.start_from_checkpoint()
        FID.set_config(device=self.device)
        gen = self.inf_train_gen()

        print("Starting Training...")
        for i in range(step_done, self.p.niters):
            for _ in range(self.p.iterD):    
                data = next(gen)
                real = data.to(self.device)
                fake, zs, ind = self.sample_g(grad=False)
                self.step_D(real, fake, zs, ind)
                #self.step_imD(real, fake)

            self.step_G()

            self.G_losses.append(errG.item())
            self.D_losses.append(errD.item())

            self.log(i, fake, real, errD_real.item(), errD_fake.item(), errG.item())
            if i%100 == 0 and i>0:
                self.fid_epoch.append(np.array(self.fid).mean())
                self.fid = []
                self.save_checkpoint(i)
        
        self.log_final(i, fake, real, errD_real.item(), errD_fake.item(), errG.item())
        print('...Done')
