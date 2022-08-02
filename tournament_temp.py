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

def get_shift(p, sort=True):
	if sort:
		alpha = 6*torch.rand(p.batch_size,2)
		alpha[(alpha < 0.5) & (alpha > 0)] = 0.5
		for i, (a1, a2) in enumerate(alpha):
			p = torch.rand(1)
			if p < 0.25:
				alpha[i,0] = -a1
				alpha[i,1] = -a2
			elif p < 0.75:
				alpha[i, 0] = -a1
		alpha = alpha.t()
		alpha = torch.sort(alpha.t())[0].t()
	else:
		alpha = ((12*torch.rand(p.batch_size,2))-6).t()
	return alpha

def sample_g(tempG, imG, z, p):
	with torch.no_grad():
		alpha = get_shift(p)
		labels = alpha[0]<alpha[1]
		z1 = tempG(z, alpha[0])
		z2 = tempG(z, alpha[1])

		zs = torch.randn(3,p.batch_size,z.shape[1], dtype=torch.float, device=p.device)
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
		im = imG(zs[0])
		im = im.reshape(-1,1,im.shape[-3],im.shape[-2],im.shape[-1])
		im1 = imG(zs[1]).reshape(-1,1,im.shape[-3],im.shape[-2],im.shape[-1])
		im2 = imG(zs[2]).reshape(-1,1,im.shape[-3],im.shape[-2],im.shape[-1])
	return torch.concat((im, im1, im2), dim=1)

def round(disc, temp_gen, im_gen, params):
	disc = disc.to(params.device)
	im_gen = im_gen.to(params.device)
	temp_gen = temp_gen.to(params.device)
	wrt = 0
	for i in range(2):
		with torch.no_grad():
			if params.ngpu > 1:
				noise = torch.randn(params.batch_size, im_gen.module.dim_z, dtype=torch.float, device=params.device)
			else:
				noise = torch.randn(params.batch_size, im_gen.dim_z, dtype=torch.float, device=params.device)

			ims = sample_g(temp_gen, im_gen, noise, params)
			f = disc(ims)
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
		for j, (tg, ig) in enumerate(gens):
			if i == j:
				continue

			wr = round(d, tg, ig, params)
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
