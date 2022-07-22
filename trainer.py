import os
import numpy as np
import pytorch_fid_wrapper as FID
import pickle
#from carbontracker.tracker import CarbonTracker

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
        #self.assessor_path = '../Assessor/Assessor/models/checkpoint_1499.pt'
        if self.p.one_disc:
            self.p.cl = False
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
        if not self.p.one_disc:
            self.imD = ImD(self.p).to(self.device)
        self.tempD = TempD(self.p).to(self.device)
        self.imG = ImG(self.p).to(self.device)
        self.tempG = TempG(self.p).to(self.device)
        
        if self.p.ngpu>1:
            if not self.p.one_disc:
                self.imD = nn.DataParallel(self.imD)
            self.tempD = nn.DataParallel(self.tempD)
            self.imG = nn.DataParallel(self.imG)
            self.tempG = nn.DataParallel(self.tempG)
            
        #state_dict = torch.load(self.assessor_path)
        #self.tempD.load_state_dict(state_dict['model'])
        if not self.p.one_disc:
            self.optimizerImD = optim.Adam(self.imD.parameters(), lr=self.p.lrImD,
                                            betas=(0., 0.9))
        self.optimizerImG = optim.Adam(self.imG.parameters(), lr=self.p.lrImG,
                                         betas=(0., 0.9))

        self.optimizerTempD = optim.Adam(self.tempD.parameters(), lr=self.p.lrTempD,
                                         betas=(0., 0.9))
        if not self.p.fixed_dir:
            self.optimizerTempG = optim.Adam(self.tempG.parameters(), lr=self.p.lrTempG,
                                            betas=(0., 0.9))
        

        self.scalerImD = GradScaler()
        self.scalerImG = GradScaler()
        self.scalerTempD = GradScaler()

        ### Make Data Generator ###
        self.generator_train = DataLoader(dataset, batch_size=self.p.batch_size, shuffle=True, num_workers=4, drop_last=True)

        ### Prep Training
        self.fixed_test_noise = None
        self.img_list = []
        self.imG_losses = []
        self.tempG_losses = []
        self.imD_losses = []
        self.tempD_losses = []
        self.fid = []
        self.fid_epoch = []

        self.cla_loss = nn.BCEWithLogitsLoss()
        self.gen_loss_scale = 0
        #self.tracker = CarbonTracker(epochs=self.p.niters, log_dir=self.p.log_dir)

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
        tempD = self.tempD_losses[-1]
        tempG_temp = self.tempG_losses[-1]

        print('[%d/%d] imD: %.2f|%.2f\ttempD: %.2f\timG: %.2f\ttempG: %.2f\tFID %.2f'
                    % (step, self.p.niters, imDr, imDf, tempD,self.imG_losses[-1], tempG_temp, self.fid[-1]))

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
            if not self.p.one_disc:
                self.imD.load_state_dict(state_dict['imD'])

            self.tempG.load_state_dict(state_dict['tempG'])
            self.tempD.load_state_dict(state_dict['tempD'])

            self.optimizerImG.load_state_dict(state_dict['optimizerImG'])
            if not self.p.one_disc:
                self.optimizerImD.load_state_dict(state_dict['optimizerImD'])

            if not self.p.fixed_dir:
                self.optimizerTempG.load_state_dict(state_dict['optimizerTempG'])
            self.optimizerTempD.load_state_dict(state_dict['optimizerTempD'])

            self.imG_losses = state_dict['lossImG']
            self.tempG_losses = state_dict['lossTempG']
            self.imD_losses = state_dict['lossImD']
            self.tempD_losses = state_dict['lossTempD']
            self.fid_epoch = state_dict['fid']
            self.gen_loss_scale = state_dict['loss_scale']
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
        'imD': self.imD.state_dict() if not self.p.one_disc else None,
        'tempG': self.tempG.state_dict(),
        'tempD': self.tempD.state_dict(),
        'optimizerImG': self.optimizerImG.state_dict(),
        'optimizerImD': self.optimizerImD.state_dict() if not self.p.one_disc else None,
        'optimizerTempG': self.optimizerTempG.state_dict() if not self.p.fixed_dir else None,
        'optimizerTempD': self.optimizerTempD.state_dict(),
        'lossImG': self.imG_losses,
        'lossTempG': self.tempG_losses,
        'lossImD': self.imD_losses,
        'lossTempD': self.tempD_losses,
        'fid': self.fid_epoch,
        'loss_scale': self.gen_loss_scale
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

    def sample_g(self, grad=False):
        with autocast():
            if grad:
                z = torch.randn(self.p.batch_size, self.p.z_size, dtype=torch.float, device=self.device)
                alpha = (12*torch.rand(self.p.batch_size,2)-6).transpose(0,1)
                if not self.p.cl:
                    alpha = torch.sort(alpa)[0]
                labels = alpha[0]<alpha[1]
                z1 = self.tempG(z, alpha[0])
                z2 = self.tempG(z, alpha[1])

                zs = torch.randn(3,self.p.batch_size,self.p.z_size, dtype=torch.float, device=self.device)
                for i, l in enumerate(labels):
                    if l and alpha[0,i]<0:
                        if alpha[1,i]<0:
                            zs[:,i] = torch.concat((
                                z1[i].reshape(1,-1),
                                z2[i].reshape(1,-1),
                                z[i].reshape(1,-1)
                            ))
                        else:
                            zs[:,i] = torch.concat((
                                z1[i].reshape(1,-1),
                                z[i].reshape(1,-1),
                                z2[i].reshape(1,-1)
                            ))
                    else:
                        zs[:,i] = torch.concat((
                            z[i].reshape(1,-1),
                            z1[i].reshape(1,-1),
                            z2[i].reshape(1,-1)
                        ))

                im = self.imG(zs[0])
                im = im.reshape(-1,1,im.shape[-3],im.shape[-2],im.shape[-1])
                im1 = self.imG(zs[1]).reshape(-1,1,im.shape[-3],im.shape[-2],im.shape[-1])
                im2 = self.imG(zs[2]).reshape(-1,1,im.shape[-3],im.shape[-2],im.shape[-1])
                ims = torch.concat((im, im1, im2), dim=1)
            else:
                with torch.no_grad():
                    z = torch.randn(self.p.batch_size, self.p.z_size, dtype=torch.float, device=self.device)
                    alpha = (12*torch.rand(self.p.batch_size,2)-6).transpose(0,1)
                    labels = alpha[0]<alpha[1]
                    z1 = self.tempG(z, alpha[0])
                    z2 = self.tempG(z, alpha[1])

                    zs = torch.randn(3,self.p.batch_size,self.p.z_size, dtype=torch.float, device=self.device)
                    for i, l in enumerate(labels):
                        if l and alpha[0,i]<0:
                            if alpha[1,i]<0:
                                zs[:,i] = torch.concat((
                                    z1[i].reshape(1,-1),
                                    z2[i].reshape(1,-1),
                                    z[i].reshape(1,-1)
                                ))
                            else:
                                zs[:,i] = torch.concat((
                                    z1[i].reshape(1,-1),
                                    z[i].reshape(1,-1),
                                    z2[i].reshape(1,-1)
                                ))
                        else:
                            zs[:,i] = torch.concat((
                                z[i].reshape(1,-1),
                                z1[i].reshape(1,-1),
                                z2[i].reshape(1,-1)
                            ))

                    im = self.imG(zs[0])
                    im = im.reshape(-1,1,im.shape[-3],im.shape[-2],im.shape[-1])
                    im1 = self.imG(zs[1]).reshape(-1,1,im.shape[-3],im.shape[-2],im.shape[-1])
                    im2 = self.imG(zs[2]).reshape(-1,1,im.shape[-3],im.shape[-2],im.shape[-1])
                    ims = torch.concat((im, im1, im2), dim=1)
        return ims, labels.reshape(-1,1).float()

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

    def step_tempD(self, real, r_label):
        for p in self.tempD.parameters():
            p.requires_grad = True
        self.tempD.zero_grad()
        with autocast():
            fake, f_label = self.sample_g()

            pred_real = self.tempD(real)
            pred_fake = self.tempD(fake)

            if self.p.cl:
                err_real = self.cla_loss(pred_real, r_label.to(self.device))
                err_fake = self.cla_loss(pred_fake, f_label.to(self.device))
            else:
                err_real = (nn.ReLU()(1.0 - pred_real)).mean()
                err_fake = (nn.ReLU()(1.0 + pred_fake)).mean()
            loss = err_real + err_fake

        self.scalerTempD.scale(loss).backward()
        self.scalerTempD.step(self.optimizerTempD)
        self.scalerTempD.update()

        for p in self.tempD.parameters():
            p.requires_grad = False

        return loss.item()

    def step_D(self, real, r_label):
        for p in self.tempD.parameters():
            p.requires_grad = True
        self.tempD.zero_grad()
        with autocast():
            fake, f_label = self.sample_g()

            disc_real, pred_real = self.tempD(real)
            disc_fake, pred_fake = self.tempD(fake)

            errD_real = (nn.ReLU()(1.0 - disc_real)).mean()
            errD_fake = (nn.ReLU()(1.0 + disc_fake)).mean()
            err_real = self.cla_loss(pred_real, r_label.to(self.device))
            err_fake = self.cla_loss(pred_fake, f_label.to(self.device))
            loss = errD_real + errD_fake + err_real + err_fake

        self.scalerTempD.scale(loss).backward()
        self.scalerTempD.step(self.optimizerTempD)
        self.scalerTempD.update()

        for p in self.tempD.parameters():
            p.requires_grad = False

        return errD_real.item(), errD_fake.item(), loss.item()

    def step_G(self):
        if not self.p.fixed_dir:
            for p in self.tempG.parameters():
                p.requires_grad = True
        for p in self.imG.parameters():
            p.requires_grad = True

        #self.tempG.zero_grad()
        self.imG.zero_grad()
        fake, label = self.sample_g(grad=True)

        with autocast():
            disc_im_fake = self.imD(fake[:,0].unsqueeze(1))
            err_im = - disc_im_fake.mean()

            pred = self.tempD(fake)
            if self.p.cl:
                err_temp = self.cla_loss(pred, label.to(self.device))
            else:
                err_temp = -pred.mean()
            
            loss = err_temp + err_im


        self.scalerImG.scale(loss).backward()
        if not self.p.fixed_dir:
            self.scalerImG.step(self.optimizerTempG)
        self.scalerImG.step(self.optimizerImG)
        self.scalerImG.update()

        if not self.p.cl:
            for p in self.tempG.parameters():
                p.requires_grad = False
        for p in self.imG.parameters():
            p.requires_grad = False

        return err_im.item(), err_temp.item(), fake[:,0]

    def train(self):
        step_done = self.start_from_checkpoint()
        FID.set_config(device=self.device)
        gen = self.inf_train_gen()

        print("Starting Training...")
        for i in range(step_done, self.p.niters):
            #self.tracker.epoch_start()
            if not self.p.one_disc:
                for _ in range(self.p.im_iter):  
                    data, labels = next(gen)
                    real = data.to(self.device)
                    errImD_real, errImD_fake = self.step_imD(real[:,0])
            else:
                errImD_real, errImD_fake = 0, 0

            for _ in range(self.p.temp_iter):
                data, labels = next(gen)
                real = data.to(self.device)
                errTempD_real = self.step_tempD(real, labels)

            errG_im, errG_temp, fake = self.step_G()

            #self.tracker.epoch_end()
            self.imG_losses.append(errG_im)
            self.tempG_losses.append(errG_temp)
            self.imD_losses.append((errImD_real, errImD_fake))
            self.tempD_losses.append(errTempD_real)

            self.log(i, fake, real)
            if i%100 == 0 and i>0:
                self.fid_epoch.append(np.array(self.fid).mean())
                self.fid = []
                self.save_checkpoint(i)
            
        
        self.log_final(i, fake, real)
        #self.tracker.stop()
        print('...Done')
