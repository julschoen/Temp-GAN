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
	state = torch.load(os.path.join(path, 'models/checkpoint.pt'))
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
		with torch.no_grad():
			with autocast():
				ims = None
				for _ in range(params.batch_size):
					if params.ngpu > 1:
						z = torch.randn(1, imG.module.dim_z, dtype=torch.float, device=params.device)
					else:
						z = torch.randn(1, imG.dim_z, dtype=torch.float, device=params.device)
					for i in range(params.time-1):
						z = torch.concat(
							(z, tempG(z[-1].unsqueeze(0)).reshape(1,-1))
						)
					print(z[0])
					print(z.mean(), z.std())
					im = imG(z)
					if ims is None:
						ims = im.reshape(1,params.time,-1,128,128)
					else:
						ims = torch.concat((ims,im.reshape(1,params.time,-1,128,128)))
		
		np.savez_compressed(os.path.join(params.log_dir,f'{model_path}_temp.npz'),x=ims.detach().cpu().numpy())

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