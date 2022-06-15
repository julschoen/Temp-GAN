import os
import numpy as np
import pytorch_fid_wrapper as FID
import pickle
from carbontracker.tracker import CarbonTracker

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
from encoder import Encoder


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

        ### load/save params
        if params.load_params:
            with open(os.path.join(params.log_dir, 'params.pkl'), 'rb') as file:
                params = pickle.load(file)
        else:
            with open(os.path.join(params.log_dir,'params.pkl'), 'wb') as file:
                pickle.dump(params, file)

        ### Make Models ###
        self.imD = ImD(self.p).to(self.device)
        self.tempD = TempD(self.p).to(self.device)
        self.imG = ImG(self.p).to(self.device)
        self.tempG = TempG(self.p).to(self.device)
        self.enc = Encoder(self.p).to(self.device)
        if self.p.ngpu>1:
            self.imD = nn.DataParallel(self.imD)
            self.tempD = nn.DataParallel(self.tempD)
            self.imG = nn.DataParallel(self.imG)
            self.tempG = nn.DataParallel(self.tempG)
            self.enc = nn.DataParallel(self.enc)

        self.optimizerImD = optim.Adam(self.imD.parameters(), lr=self.p.lrImD,
                                         betas=(0., 0.9))
        self.optimizerImG = optim.Adam(self.imG.parameters(), lr=self.p.lrImG,
                                         betas=(0., 0.9))

        self.optimizerTempD = optim.Adam(self.tempD.parameters(), lr=self.p.lrTempD,
                                         betas=(0., 0.9))
        self.optimizerTempG = optim.Adam(self.tempG.parameters(), lr=self.p.lrTempG,
                                         betas=(0., 0.9))
        self.optimizerEnc = optim.Adam(self.enc.parameters(), lr=self.p.lrImG,
                                         betas=(0., 0.9))

        self.scalerImD = GradScaler()
        self.scalerImG = GradScaler()
        self.scalerTempD = GradScaler()
        self.scalerTempG = GradScaler()
        self.scalerEnc = GradScaler()

        ### Make Data Generator ###
        self.generator_train = DataLoader(dataset, batch_size=self.p.batch_size, shuffle=True, num_workers=4, drop_last=True)

        ### Prep Training
        self.fixed_test_noise = None
        self.img_list = []
        self.imG_losses = []
        self.tempG_losses = []
        self.imD_losses = []
        self.tempD_losses = []
        self.Rec_losses = []
        self.fid = []
        self.fid_epoch = []

        self.reg_loss = nn.MSELoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=0.5)
        self.tracker = CarbonTracker(epochs=self.p.niters, log_dir=self.p.log_dir)

    def inf_train_gen(self):
        while True:
            for data in self.generator_train:
                yield data
        
    def log_train(self, step, fake, real):
        with torch.no_grad():
            self.fid.append(
                FID.fid(
                    torch.reshape(fake.to(torch.float32), (-1,1,128,128)).expand(-1,3,-1,-1), 
                    real_images=torch.reshape(real.to(torch.float32), (-1,1,128,128)).expand(-1,3,-1,-1)
                    )
                )
        imDr, imDf = self.imD_losses[-1]
        tempDr, tempDf = self.tempD_losses[-1]
        tempG_im, tempG_temp = self.tempG_losses[-1]
        err_rec, err_gan = self.Rec_losses[-1]

        print('[%d/%d] imD: %.2f|%.2f\ttempD: %.2f|%.2f\tRec: %.2f|%.2f\timG: %.2f\ttempG (im|temp): %.2f|%.2f\tFID %.2f'
                    % (step, self.p.niters, imDr, imDf, tempDr, tempDf, err_rec, err_gan,\
                        self.imG_losses[-1], tempG_im, tempG_temp, self.fid[-1]))

    def log_interpolation(self, step):
        noise = torch.randn(self.p.batch_size, self.p.z_size, dtype=torch.float, device=self.device)
        if self.fixed_test_noise is None:
            self.fixed_test_noise = noise.clone()
    
        with torch.no_grad():
            fake = self.imG(self.fixed_test_noise).detach().cpu()
        torchvision.utils.save_image(
            vutils.make_grid(torch.reshape(fake, (-1,1,128,128)), padding=2, normalize=True)
            , os.path.join(self.images_dir, f'{step}.png'))

    def start_from_checkpoint(self):
        step = 0
        files = [f for f in os.listdir(self.models_dir)]
        if len(files) < 2:
            checkpoint = os.path.join(self.models_dir, 'checkpoint.pt')
        else:
            files.remove('checkpoint.pt')
            files = sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0]))
            checkpoint = os.path.join(self.models_dir, files[-1])
        if os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint)
            step = state_dict['step']
            self.imG.load_state_dict(state_dict['imG'])
            self.imD.load_state_dict(state_dict['imD'])

            self.tempG.load_state_dict(state_dict['tempG'])
            self.tempD.load_state_dict(state_dict['tempD'])

            self.enc.load_state_dict(state_dict['enc'])

            self.optimizerImG.load_state_dict(state_dict['optimizerImG'])
            self.optimizerImD.load_state_dict(state_dict['optimizerImD'])

            self.optimizerTempG.load_state_dict(state_dict['optimizerTempG'])
            self.optimizerTempD.load_state_dict(state_dict['optimizerTempD'])

            self.optimizerEnc.load_state_dict(state_dict['optimizerEnc'])

            self.imG_losses = state_dict['lossImG']
            self.tempG_losses = state_dict['lossTempG']
            self.imD_losses = state_dict['lossImD']
            self.tempD_losses = state_dict['lossTempD']
            self.Rec_losses = state_dict['lossRec']
            self.fid_epoch = state_dict['fid']
            print('starting from step {}'.format(step))
        return step

    def save_checkpoint(self, step):
        if step < self.p.niters - 1001:
            name = 'checkpoint.pt'
        else:
            name = f'checkpoint_{step}.pt'

        torch.save({
        'step': step,
        'imG': self.imG.state_dict(),
        'imD': self.imD.state_dict(),
        'tempG': self.tempG.state_dict(),
        'tempD': self.tempD.state_dict(),
        'enc': self.enc.state_dict(),
        'optimizerImG': self.optimizerImG.state_dict(),
        'optimizerImD': self.optimizerImD.state_dict(),
        'optimizerTempG': self.optimizerTempG.state_dict(),
        'optimizerTempD': self.optimizerTempD.state_dict(),
        'optimizerEnc': self.optimizerEnc.state_dict(),
        'lossImG': self.imG_losses,
        'lossTempG': self.tempG_losses,
        'lossImD': self.imD_losses,
        'lossTempD': self.tempD_losses,
        'lossRec': self.Rec_losses,
        'fid': self.fid_epoch,
        }, os.path.join(self.models_dir, name))

    def log(self, step, fake, real):
        if step % self.p.steps_per_log == 0:
            self.log_train(step, fake, real)

        if step % self.p.steps_per_img_log == 0:
            self.log_interpolation(step)

    def log_final(self, step, fake, real):
        self.log_train(step, fake, real)
        self.log_interpolation(step)
        self.save_checkpoint(step)

    def sample_g(self):
        ims = None
        zs = None
        inds = None
        with autocast():
            z = torch.randn(self.p.batch_size, self.p.z_size, dtype=torch.float, device=self.device)
            one = torch.ones(self.p.batch_size).to(self.p.device).reshape(-1,1)
            two = torch.Tensor([2.]*self.p.batch_size).to(self.p.device).reshape([-1,1])
            shift1 = self.tempG(one)
            shift2 = self.tempG(two)
            im = self.imG(z).reshape(-1,1,64,128,128)
            im1 = self.imG(z+shift1).reshape(-1,1,64,128,128)
            im2 = self.imG(z+shift2).reshape(-1,1,64,128,128)
            ims = torch.concat((im, im1, im2), dim=1)
        return ims

    def step_imD(self, real):
        for p in self.imD.parameters():
            p.requires_grad = True
        
        self.imD.zero_grad()
        with autocast():
            z = torch.randn(real.shape[0], self.p.z_size, dtype=torch.float, device=self.device)
            fake = self.imG(z)
            disc_fake = self.imD(fake)
            disc_real = self.imD(real.unsqueeze(1))
            errD_real = (nn.ReLU()(1.0 - disc_real)).mean()
            errD_fake = (nn.ReLU()(1.0 + disc_fake)).mean()
            
            errImD = errD_fake + errD_real
        self.scalerImD.scale(errImD).backward()
        self.scalerImD.step(self.optimizerImD)
        self.scalerImD.update()

        for p in self.imD.parameters():
            p.requires_grad = False

        return errD_real.item(), errD_fake.item()

    def step_tempD(self, real):
        for p in self.tempD.parameters():
            p.requires_grad = True
        self.tempD.zero_grad()
        with autocast():
            fake = self.sample_g()
            disc_fake = self.tempD(fake)
            disc_real = self.tempD(real)
            errD_real = (nn.ReLU()(1.0 - disc_real)).mean()
            errD_fake = (nn.ReLU()(1.0 + disc_fake)).mean()
            errTempD = errD_fake + errD_real
        self.scalerTempD.scale(errTempD).backward()
        self.scalerTempD.step(self.optimizerTempD)
        self.scalerTempD.update()

        for p in self.tempD.parameters():
            p.requires_grad = False

        return errD_real.item(), errD_fake.item()

    def step_imG(self):
        for p in self.imG.parameters():
            p.requires_grad = True

        self.imG.zero_grad()
        with autocast():
            z = torch.randn(self.p.batch_size, self.p.z_size, dtype=torch.float, device=self.device)
            fake = self.imG(z)
            disc_fake = self.imD(fake)
            errImG = - disc_fake.mean()

        self.scalerImG.scale(errImG).backward()
        self.scalerImG.step(self.optimizerImG)
        self.scalerImG.update()

        for p in self.imG.parameters():
            p.requires_grad = False

        return errImG.item(), fake

    def step_tempG(self):
        for p in self.tempG.parameters():
            p.requires_grad = True

        self.tempG.zero_grad()
        fake = self.sample_g()

        with autocast():
            disc_temp_fake = self.imD(fake[:,1+torch.randint(2, ())].unsqueeze(1))
            errTempG = - disc_temp_fake.mean()

        self.scalerTempG.scale(errTempG).backward()
        self.scalerTempG.step(self.optimizerTempG)
        self.scalerTempG.update()

        for p in self.tempG.parameters():
            p.requires_grad = False

        return errTempG.item()

    def step_Enc(self, real):
        for p in self.imG.parameters():
            p.requires_grad = True
        for p in self.enc.parameters():
            p.requires_grad = True

        self.enc.zero_grad()
        self.imG.zero_grad()

        with autocast():
            zs = self.enc(real.unsqueeze(1))
            rec = self.imG(zs)
            errGAN = -self.imD(rec).mean()
            errRec = torch.log(self.reg_loss(rec.squeeze(),real.squeeze()))
            loss = errRec + errGAN * 0.1

        self.scalerEnc.scale(loss).backward()
        self.scalerEnc.step(self.optimizerEnc)
        self.scalerEnc.step(self.optimizerImG)
        self.scalerEnc.update()

        for p in self.imG.parameters():
            p.requires_grad = False
        for p in self.enc.parameters():
            p.requires_grad = False

        return errRec.item(), errGAN.item()

    def step_TripletD(self, real):
        for p in self.tempD.parameters():
            p.requires_grad = True

        self.tempD.zero_grad()

        with autocast():
            real = real.reshape(-1,3,1,real.shape[-3],real.shape[-2],real.shape[-1])
            r1, r2, r3 = real[:,0], real[:,1], real[:,2]

            h1 = self.tempD(r1)
            h2 = self.tempD(r2)
            h3 = self.tempD(r3)

            l1 = self.triplet_loss(h1,h2,h3)
            l2 = self.triplet_loss(h3,h2,h1)
            loss = l1+l2

        self.scalerTempD.scale(loss).backward()
        self.scalerTempD.step(self.optimizerTempD)
        self.scalerTempD.update()

        for p in self.tempD.parameters():
            p.requires_grad = False

        return loss.item()

    def step_TripletG(self):
        for p in self.tempG.parameters():
            p.requires_grad = True
        for p in self.imG.parameters():
            p.requires_grad = True

        self.tempG.zero_grad()
        self.imG.zero_grad()
        fake = self.sample_g()

        with autocast():
            #fake = fake.reshape(-1,3,1,fake.shape[-3],fake.shape[-2],fake.shape[-1])
            #f1, f2, f3 = fake[:,0], fake[:,1], fake[:,2]

            #h1 = self.tempD(fake)
            #h2 = self.tempD(f2)
            #h3 = self.tempD(f3)

            disc_temp_fake = self.tempD(fake)
            errTempG = - disc_temp_fake.mean()

            #l1 = self.triplet_loss(h1,h2,h3)
            #l2 = self.triplet_loss(h3,h2,h1)
            #loss = l1+l2

        self.scalerTempG.scale(errTempG).backward()
        self.scalerTempG.step(self.optimizerTempG)
        self.scalerTempG.step(self.optimizerImG)
        self.scalerTempG.update()

        for p in self.tempG.parameters():
            p.requires_grad = False
        for p in self.imG.parameters():
            p.requires_grad = False

        return errTempG.item()

    def train(self):
        step_done = self.start_from_checkpoint()
        FID.set_config(device=self.device)
        gen = self.inf_train_gen()

        print("Starting Training...")
        for i in range(step_done, self.p.niters):
            self.tracker.epoch_start()
            for _ in range(self.p.im_iter):
                for _ in range(self.p.iterD):  
                    data, _ = next(gen)
                    real = data.to(self.device)
                    errImD_real, errImD_fake = self.step_imD(real[:,0])
                errImG, fake = self.step_imG()
                err_rec, err_gan = self.step_Enc(real[:,0])
                

            for _ in range(self.p.temp_iter):
                errTempD_real, errTempD_fake = self.step_tempD(real)
                errTempG_im = self.step_tempG()
                errTempG_temp = self.step_TripletG()
            self.tracker.epoch_end()
            self.imG_losses.append(errImG)
            self.tempG_losses.append((errTempG_im, errTempG_temp))
            self.imD_losses.append((errImD_real, errImD_fake))
            self.tempD_losses.append((errTempD_real, errTempD_fake))
            self.Rec_losses.append((err_rec, err_gan))

            self.log(i, fake, real)
            if i%100 == 0 and i>0:
                self.fid_epoch.append(np.array(self.fid).mean())
                self.fid = []
                self.save_checkpoint(i)
            
        
        self.log_final(i, fake, real)
        self.tracker.stop()
        print('...Done')

