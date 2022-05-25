import numpy as np
import os
import argparse
import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.autograd import Variable

from image_gen import Generator as ImG
from temp_gen import Generator as TempG
from image_disc import Discriminator as ImD
from temp_data_handler import DATA

def load_models(path, ngpu):
	with open(os.path.join(path, 'params.pkl'), 'rb') as file:
		params = pickle.load(file)
	
	imG = ImG(params)
	tempG = TempG(params)
	imD = ImD(params)

	if ngpu > 1:
		imG = nn.DataParallel(imG)
		tempG = nn.DataParallel(tempG)
		imD = nn.DataParallel(imD)
	state = torch.load(os.path.join(path, 'models/checkpoint.pt'))
	imG.load_state_dict(state['imG'])
	tempG.load_state_dict(state['tempG'])
	imD.load_state_dict(state['imD'])

	return imG.to(params.device), tempG.to(params.device), imD.to(params.device)

def generate_ims(netG, params, save_name, noise=None):
	if noise is None:
		if params.ngpu > 1:
			noise = torch.randn(params.batch_size, netG.module.dim_z, dtype=torch.float, device=params.device)
		else:
			noise = torch.randn(params.batch_size, netG.dim_z, dtype=torch.float, device=params.device)
	with torch.no_grad():
		with autocast():
			ims = netG(noise)
	ims = ims.detach().cpu().numpy()
	np.savez_compressed(os.path.join(params.log_dir, save_name), x=ims)

def get_embedding(ims, imD):
	with torch.no_grad():
		with autocast():
			_, zs = imD(ims.unsqueeze(1))
	return zs

def reverse_z(netG, ims, params, niter=5000, lr=0.01):
	mse_loss = nn.MSELoss().to(params.device)
	if params.ngpu > 1:
		z_approx = torch.randn(ims.shape[0], netG.module.dim_z,dtype=torch.float, device=params.device)
	else:
		z_approx = torch.randn(ims.shape[0], netG.dim_z,dtype=torch.float, device=params.device)
	z_approx = Variable(z_approx)
	z_approx.requires_grad = True

	optimizer_approx = torch.optim.Adam([z_approx], lr=lr,betas=(0.5, 0.999))
	netG.eval()
	with autocast():
		for i in range(niter):
			g_z_approx = netG(z_approx)
			mse_g_z = mse_loss(g_z_approx.squeeze(), ims)

			if i % 500 == 0:
				print("[Iter {}] mse_g_z: {}".format(i, mse_g_z.item()))

			optimizer_approx.zero_grad()
			mse_g_z.backward()
			optimizer_approx.step()
	return z_approx


def eval(params):
	dataset = DATA(path=params.data_path)
	print(dataset.__len__())
	generator = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=4)
	os.makedirs(params.log_dir, exist_ok=True)
	for model_path in params.model_log:
		print(model_path)
		imG, tempG, imD = load_models(model_path, params.ngpu)
		generate_ims(imG, params, f'random_gen_{model_path}.npz')
		for _, (data, _) in enumerate(generator):
			data = data[:,0].to(params.device)
			zs = get_embedding(data, imD)
			rev_zs = reverse_z(imG, data, params)
			generate_ims(imG, params, f'rec_gen_{model_path}.npz', noise=zs)
			generate_ims(imG, params, f'rev_rec_gen_{model_path}.npz', noise=rev_zs)
			np.savez_compressed(os.path.join(params.log_dir, f'rec_real_{model_path}.npz'), x=data.detach().cpu().numpy())
			break

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--data_path', type=str, default='data/test_pat.npz',help='Path to data.')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('-l', '--model_log', action='append', type=str, required=True, help='Model log directories to evaluate')
	params = parser.parse_args()
	eval(params)

if __name__ == '__main__':
	main()
