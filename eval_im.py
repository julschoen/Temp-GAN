import numpy as np
import os
import argparse
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.cuda.amp import autocast

from eval_utils import *
from image_gen import Generator as ImG
from temp_data_handler import DATA, Data4D, DataLIDC, DataCBCT

def load_gen(path, ngpu):
	with open(os.path.join(path, 'params.pkl'), 'rb') as file:
		params = pickle.load(file)
	
	netG = ImG(params)

	if ngpu > 1:
		netG = nn.DataParallel(netG)
	state = torch.load(os.path.join(path, 'models/checkpoint_4999.pt'))
	netG.load_state_dict(state['imG'])

	return netG

def eval(params):
	#dataset = DataLIDC(path=params.data_path, triplet=True)
	dataset = Data4D(path='../Data/4dct_clean/test_pat.npz', shift=False)
	#dataset = DataCBCT(path='../Data/cbct/test_pat.npz', shift=False)
	print(dataset.__len__())
	generator = DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=4)
	os.makedirs(params.log_dir, exist_ok=True)
	for model_path in params.model_log:
		print(model_path)
		netG = load_gen(model_path, params.ngpu).to(params.device)
		fids_ax = []
		fids_cor = []
		fids_sag = []
		large_data = None
		large_fake = None
		with torch.no_grad():
			for i, (data, _) in enumerate(generator):
				x1 = data
				#x1 = x1[:,np.random.choice([0,1,2])].unsqueeze(1)
				if params.ngpu > 1:
					noise = torch.randn(x1.shape[0], netG.module.dim_z, dtype=torch.float, device=params.device)
				else:
					noise = torch.randn(x1.shape[0], netG.dim_z, dtype=torch.float, device=params.device)
				x2 = netG(noise)
				fa, fc, fs = fid(x1, x2, params.device)
				fids_ax.append(fa)
				fids_cor.append(fc)
				fids_sag.append(fs)
		fids_ax = np.array(fids_ax)
		fids_cor = np.array(fids_cor)
		fids_sag = np.array(fids_sag)
		print(f'FID ax: {fids_ax.mean():.1f}+-{fids_ax.std():.1f}'+
			f' cor: {fids_cor.mean():.1f}+-{fids_cor.std():.1f}'+
			f' sag: {fids_sag.mean():.1f}+-{fids_sag.std():.1f}')
		np.savez_compressed(os.path.join(params.log_dir,f'{model_path}_stats.npz'), fid_ax=fids_ax, fid_cor=fids_cor, fid_sag=fids_sag)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--data_path', type=str, default='../3D-GAN/train_lidc_128.npz',help='Path to data.')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('-l', '--model_log', action='append', type=str, required=True, help='Model log directories to evaluate')
	params = parser.parse_args()

	eval(params)

if __name__ == '__main__':
	main()
