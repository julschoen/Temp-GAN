import numpy as np
import os
import argparse
import torch
from torch.nn.functional import interpolate
import nibabel as nib

def main():
	print(np.load('data/test_pat.npz')['x'])
	print(np.load('data/train_pat.npz')['x'])

if __name__ == '__main__':
	main()
