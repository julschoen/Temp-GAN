import numpy as np
import os
import argparse
import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from image_gen import Generator
from image_disc import Discriminator

def load_model(path, ngpu):
    with open(os.path.join(path, 'params.pkl'), 'rb') as file:
        params = pickle.load(file)

    netG = Generator(params)
    netD = Discriminator(params)

    if ngpu > 1:
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    state = torch.load(os.path.join(path, 'models/checkpoint.pt'))
    netG.load_state_dict(state['imG'])
    netD.load_state_dict(state['imD'])

    return netD, netG

def round(disc, gen, params):
	disc = disc.to(params.device)
	gen = gen.to(params.device)
	wrt = 0
	for i in range(2):
		with torch.no_grad():
			if params.ngpu > 1:
				noise = torch.randn(params.batch_size, gen.module.dim_z,
						1, 1, 1, dtype=torch.float, device=params.device)
			else:
				noise = torch.randn(params.batch_size, gen.dim_z,
						1, 1, 1, dtype=torch.float, device=params.device)
			f = disc(gen(noise))
			wrt += (f > 0).sum().item()

	disc, gen = disc.cpu(), gen.cpu()

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
		d,g = load_model(model, params.ngpu)
		discs.append(d)
		gens.append(g)

	tournament(discs, gens, params)

if __name__ == '__main__':
	main()
