import numpy as np
import os
import argparse
import torch
from torch.nn.functional import interpolate
import nibabel as nib

def main():
	test = np.load('data/test_pat.npz')['x']
	train = np.load('data/train_pat.npz')['x']

	new_test = []
	new_train = []

	for t in test:
		if t.startswith('data/'):
			new_test.append(t[5:])
		else:
			new_test.append(t)

	for t in train:
		if t.startswith('data/'):
			new_train.append(t[5:])
		else:
			new_train.append(t)

	print(new_train)
	print(new_test)

	np.savez_compressed('data/test_pat.npz', x=np.array(new_test))
	np.savez_compressed('data/train_pat.npz', x=np.array(new_train))

if __name__ == '__main__':
	main()
