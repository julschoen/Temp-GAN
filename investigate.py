import numpy as np
import os
import argparse
import torch
from torch.nn.functional import interpolate

def main():
	path = 'data'
	dirs = np.load(os.path.join(path, 'test_pat.npz'))['x']

	new_dirs = []
	for d in dirs:
		files = sorted([f for f in os.listdir(os.path.join(path, d)) if f.endswith('.npz')],
					key=lambda x: int(x[:-4]))
		new_dirs.append(d)
		if len(files) > 1:
			for i, f in enumerate(files):
				x = np.load(os.path.join(path,d,f))['x']
				if i == 0:
					np.savez_compressed(os.path.join(path, d, '0.npz'), x=x)
				else:
					new_path = os.path.join(path, d+f'_{i}')
					os.makedirs(new_path, exist_ok=True)
					np.savez_compressed(os.path.join(new_path, '0.npz'), x=x)
					new_dirs.append(new_path)

	dirs = np.array(new_dirs)
	print(dirs.shape)
	np.savez_compressed(os.path.join(path, 'test_pat.npz'), x=dirs)

if __name__ == '__main__':
	main()
