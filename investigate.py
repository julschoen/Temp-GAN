import numpy as np
import os
import nibabel as nib
import argparse
import torch
from torch.nn.functional import interpolate

def main():
	path = 'data'
	dirs = np.load(os.path.join(path, 'test_pat.npz'))['x']

	data = None
	for d in dirs:
		files = sorted([f for f in os.listdir(os.path.join(path, d)) if f.endswith('.npz')],
					key=lambda x: int(x[:-4]))

		for f in files:
			x = np.load(os.path.join(path,d,f))['x']
			
			if data is None:
				data = x
			else:
				data = np.concatenate((data, x))

	print(data.shape)
	np.savez_compressed('test_cbct.npz', x=data)

if __name__ == '__main__':
	main()
