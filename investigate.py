import numpy as np
import os
import nibabel as nib
import argparse
import torch
from torch.nn.functional import interpolate

def main():
	path = 'data'

	dirs = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))],
		key=lambda x: int(x))
	for d in dirs:
		files = sorted([f for f in os.listdir(os.data_path.join(path, d)) if f.endswith('.npz')],
					key=lambda x: int(x[:-4]))

		if len(files) > 1:
			print(d)
			for f in files:
				x = np.load(os.path.join(path,d,f))['x']
				print(f'File {f} len {x.shape[0]}')

if __name__ == '__main__':
	main()
