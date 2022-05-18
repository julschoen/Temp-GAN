import numpy as np
import os
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
		print(d,files)
		shapes = []
		for f in files:
			x = np.load(os.path.join(path,d,f))['x']
			shapes.append(x.shape)
		print(shapes)

if __name__ == '__main__':
	main()
