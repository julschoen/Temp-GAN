import numpy as np
import os
import argparse
import torch
from torch.nn.functional import interpolate
import nibabel as nib

def main():
	path = '../../../media/7tb_encrypted/julians_project/anon_images_updated'

	dirs = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))],
		key=lambda x: int(x))
	for d in dirs:
		subdirs = sorted([f for f in os.listdir(os.path.join(path, d)) 
									if os.path.isdir(os.path.join(path,d,f))],
				key=lambda x: int(x))
		for sub in subdirs:
			files = sorted([f for f in os.listdir(os.path.join(path,d,sub)) if f.startswith('cbct')],
							key=lambda x: int(x[4:5]) if x[4:6].endswith('_') else int(x[4:6]))
			if len(files)>1:
				for f in files:
					if not f.endswith('.gz'):
						continue
					img = nib.load(os.path.join(path,d,sub,f))
					print(img.header)

if __name__ == '__main__':
	main()
