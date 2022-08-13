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
from encoder import Encoder
from image_disc import Discriminator as ImD
from temp_data_handler import DATA

def load_models(path, ngpu):
	with open(os.path.join(path, 'params.pkl'), 'rb') as file:
		params = pickle.load(file)
	
	imG = ImG(params)
	tempG = TempG(params)

	if ngpu > 1:
		imG = nn.DataParallel(imG)
		tempG = nn.DataParallel(tempG)
	state = torch.load(os.path.join(path, 'models/checkpoint_4999.pt'))
	imG.load_state_dict(state['imG'])
	tempG.load_state_dict(state['tempG'])

	return imG.to(params.device), tempG.to(params.device)

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

def embed_check(netG, params):
	if params.ngpu > 1:
		z1, z2 = torch.randn(2, netG.module.dim_z,dtype=torch.float, device=params.device)
		z_size = netG.module.dim_z
	else:
		z1, z2 = torch.randn(2, netG.dim_z,dtype=torch.float, device=params.device)
		z_size = netG.dim_z

	z12 = z2-z1
	mags = torch.linspace(0,1,10)

	zs = torch.randn(10,z_size, dtype=torch.float, device=params.device)
	for i, m in enumerate(mags):
		zs[i] = z1 + z12*m

	return zs



def eval(params):
	for model_path in params.model_log:
		print(model_path)
		imG, tempG = load_models(model_path, params.ngpu)
		zs = embed_check(imG, params)
		generate_ims(imG, params, f'embed_{model_path}.npz', noise=zs)
		#generate_ims(imG, params, f'rev_rec_gen_{model_path}.npz', noise=rev_zs)
		#np.savez_compressed(os.path.join(params.log_dir, f'rec_real_{model_path}.npz'), x=data.detach().cpu().numpy())

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
