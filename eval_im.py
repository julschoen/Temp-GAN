import numpy as np
import os
import argparse
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.cuda.amp import autocast

from eval_utils import *
from image_gen import Generator as ImG
from temp_data_handler import DATA

def load_gen(path, ngpu):
	with open(os.path.join(path, 'params.pkl'), 'rb') as file:
		params = pickle.load(file)
	
	netG = ImG(params)

	if ngpu > 1:
		netG = nn.DataParallel(netG)
	state = torch.load(os.path.join(path, 'models/checkpoint.pt'))
	netG.load_state_dict(state['imG'])

	return netG

def eval(params):
	dataset = DATA(path=params.data_path)
	print(dataset.__len__())
	generator = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=4)
	fid_model = get_fid_model(params.fid_checkpoint).to(params.device)
	if params.ngpu > 1:
		fid_model = nn.DataParallel(fid_model)
	os.makedirs(params.log_dir, exist_ok=True)
	for model_path in params.model_log:
		print(model_path)
		netG = load_gen(model_path, params.ngpu).to(params.device)
		ssims = []
		psnrs = []
		fids = []
		fids_ax = []
		fids_cor = []
		fids_sag = []
		large_data = None
		large_fake = None
		with torch.no_grad():
			with autocast():
				for i, (data, _) in enumerate(generator):
					x1 = data[:,0].unsqueeze(dim=1)
					if params.ngpu > 1:
						noise = torch.randn(data.shape[0], netG.module.dim_z, dtype=torch.float, device=params.device)
					else:
						noise = torch.randn(data.shape[0], netG.dim_z, dtype=torch.float, device=params.device)
					x2 = netG(noise)
					if i % 16 == 0 and i>0:
						s,p,f = ssim(large_data,large_fake), psnr(large_data,large_fake),fid_3d(fid_model, large_data, large_fake)
						ssims.append(s)
						psnrs.append(p)
						fids.append(f)
						large_data = None
						large_fake = None
					else:
						if large_data is not None and large_fake is not None:
							large_data = torch.concat((large_data, x1.cpu()))
							large_fake = torch.concat((large_fake, x2.cpu()))
						else:
							large_data = x1.cpu()
							large_fake = x2.cpu()
					if i%5 == 0:
						fa, fc, fs = fid(x1, x2, params.device)
						fids_ax.append(fa)
						fids_cor.append(fc)
						fids_sag.append(fs)
			

		ssims = np.array(ssims)
		psnrs = np.array(psnrs)
		fids = np.array(fids)
		fids_ax = np.array(fids_ax)
		fids_cor = np.array(fids_cor)
		fids_sag = np.array(fids_sag)
		print(f'SSIM: {ssims.mean():.4f}+-{ssims.std():.4f}'+ 
			f'\tPSNR: {psnrs.mean():.4f}+-{psnrs.std():.4f}'+
			f'\tFID ax: {fids_ax.mean():.4f}+-{fids_ax.std():.4f}'+
			f'\tFID cor: {fids_cor.mean():.4f}+-{fids_cor.std():.4f}'+
			f'\tFID sag: {fids_sag.mean():.4f}+-{fids_sag.std():.4f}'+
			f'\t3d-FID: {fids.mean():.4f}+-{fids.std():.4f}')
		np.savez_compressed(os.path.join(params.log_dir,f'{model_path}_stats.npz'),
			ssim = ssims, psnr = psnrs, fid = fids, fid_ax=fids_ax, fid_cor=fids_cor, fid_sag=fids_sag)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--data_path', type=str, default='data2/test_pat.npz',help='Path to data.')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('-l', '--model_log', action='append', type=str, required=True, help='Model log directories to evaluate')
	parser.add_argument('--fid_checkpoint', type=str, default='resnet_50.pth', help='Path to pretrained MedNet')
	params = parser.parse_args()
	eval(params)

if __name__ == '__main__':
	main()
