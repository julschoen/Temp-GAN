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
	print(dirs)
	print(len(dirs))


if __name__ == '__main__':
	main()
