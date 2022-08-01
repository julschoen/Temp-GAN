import numpy as np
import os
import argparse
import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from image_gen import Generator as ImageG
from temp_gen import Generator as TempG
from temp_disc import Discriminator

def load_model(path, ngpu):
    with open(os.path.join(path, 'params.pkl'), 'rb') as file:
        params = pickle.load(file)

    imG = ImageG(params)
    tempG = TempG(params)
    netD = Discriminator(params)

    if ngpu > 1:
        imG = nn.DataParallel(imG)
        tempG = nn.DataParallel(tempG)
        netD = nn.DataParallel(netD)

    state = torch.load(os.path.join(path, 'models/checkpoint.pt'))
    imG.load_state_dict(state['imG'])
    tempG.load_state_dict(state['tempG'])
    netD.load_state_dict(state['tempD'])

    return netD, tempG, imG

def sample_g(tempG, imG):
    with torch.no_grad():
        if params.ngpu > 1:
				z = torch.randn(params.batch_size, gen.module.dim_z, dtype=torch.float, device=params.device)
			else:
				z = torch.randn(params.batch_size, gen.dim_z, dtype=torch.float, device=params.device)
        alpha = torch.sort((12*torch.rand(params.batch_size,2)-6))[0].transpose(0,1)
        labels = alpha[0]<alpha[1]
        z1 = self.tempG(z, alpha[0])
        z2 = self.tempG(z, alpha[1])

        zs = torch.randn(params.batch_size,z.shape[1], dtype=torch.float, device=self.device)
        for i, _ in enumerate(labels):
            if alpha[1,i]<0 and alpha[0,i]<0:
                zs[:,i] = torch.concat((
                    z1[i].reshape(1,-1),
                    z2[i].reshape(1,-1),
                    z[i].reshape(1,-1)
                ))
            elif alpha[0,i]<0:
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

def round(disc, im_gen, temp_gen, params):
	disc = disc.to(params.device)
	im_gen = im_gen.to(params.device)
	temp_gen = temp_gen.to(params.device)
	wrt = 0
	for i in range(2):
		with torch.no_grad():
			if params.ngpu > 1:
				noise = torch.randn(params.batch_size, gen.module.dim_z, dtype=torch.float, device=params.device)
			else:
				noise = torch.randn(params.batch_size, gen.dim_z, dtype=torch.float, device=params.device)

			f = disc(gen(noise))
			wrt += (f > 0).sum().item()

	disc, im_gen, temp_gen = disc.cpu(), im_gen.cpu(), temp_gen.cpu()

	wrt =wrt/(params.batch_size*2)
	return wrt

def tournament(discs, gens, params):
	names = params.model_log
	res = {}

	for n in names:
		res[n] = []
	for i, d in enumerate(discs):
		for j, g in enumerate(gens):
			if i == j:
				continue

			wr = round(d, g, params)
			res[names[j]].append(wr)

	print('------------- Tournament Results -------------')
	for n in names:
		g = res[n]
		wr = np.mean(g)
		print(f'G of {n} with Mean Win Rate of {wr:.4f}')

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('-l', '--model_log', action='append', type=str, required=True, help='Model log directories to evaluate')
	params = parser.parse_args()

	discs, gens = [], []
	for model in params.model_log:
		d, temp_g, im_g = load_model(model, params.ngpu)
		discs.append(d)
		gens.append((temp_g, im_g))

	tournament(discs, gens, params)

if __name__ == '__main__':
	main()