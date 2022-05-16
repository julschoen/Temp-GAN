import numpy as np
import os
import nibabel as nib
import argparse
import torch
from torch.nn.functional import interpolate

def main():
	path = 'data'
	dirs = np.load(os.path.join(path, 'train_pat.npz'))['x']

	data = None
	for d in dirs:
		files = sorted([f for f in os.listdir(os.path.join(path, d)) if f.endswith('.npz')],
					key=lambda x: int(x[:-4]))

		for f in files:
			x = np.load(os.path.join(path,d,f))['x']
			print(x.shape)
			x = x.reshape(-1,1,128,128,128)
			print(x.shape)
			break

if __name__ == '__main__':
	main()
