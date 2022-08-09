import numpy as np
import os
import argparse
import pickle
import torch.nn as nn
from torch.cuda.amp import autocast
import torch

from image_gen import Generator as ImG
from temp_gen import Generator as TempG

def load_gen(path, ngpu):
	with open(os.path.join(path, 'params.pkl'), 'rb') as file:
		params = pickle.load(file)
	
	imG = ImG(params)
	tempG = TempG(params)

	if ngpu > 1:
		imG = nn.DataParallel(imG)
		tempG = nn.DataParallel(tempG)
	state = torch.load(os.path.join(path, 'models/checkpoint_9999.pt'))
	imG.load_state_dict(state['imG'])
	tempG.load_state_dict(state['tempG'])

	return imG, tempG

def eval(params):
	os.makedirs(params.log_dir, exist_ok=True)
	for model_path in params.model_log:
		print(model_path)
		imG, tempG = load_gen(model_path, params.ngpu)
		imG = imG.to(params.device)
		tempG = tempG.to(params.device)
		shifts_r = 12.
		with torch.no_grad():
			if params.ngpu > 1:
				z = torch.randn(params.batch_size, imG.module.dim_z, dtype=torch.float, device=params.device)
			else:
				z = torch.randn(params.batch_size, imG.dim_z, dtype=torch.float, device=params.device)
			np.arange(-shifts_r, shifts_r + 1e-9, shifts_r / 10)
			alpha = torch.arange(-shifts_r, shifts_r + 1e-9, shifts_r / 8).repeat(params.batch_size).reshape(params.batch_size,17).t()

			im = None
			for a in alpha:
				im1 = imG(tempG(z,a)).unsqueeze(1)
				if im is None:
					im = im1
				else:
					im = torch.concat((im, im1), dim=1)
		
		np.savez_compressed(os.path.join(params.log_dir,f'{model_path}2_temp.npz'),x=im.detach().cpu().numpy())

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--time', type=int, default=5, help='Number of time steps to generate')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('-l', '--model_log', action='append', type=str, required=True, help='Model log directories to evaluate')
	params = parser.parse_args()
	eval(params)

if __name__ == '__main__':
	main()