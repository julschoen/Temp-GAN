import numpy as np
import os
import nibabel as nib
import argparse
import torch
from torch.nn.functional import interpolate

def process(path):
	files = sorted([f for f in os.listdir(path) if f.startswith('cbct')],
		key=lambda x: int(x[4:5]) if x[4:6].endswith('_') else int(x[4:6]))
	ims = None
	for f in files:
		img = nib.load(os.path.join(path,f))
		img = torch.Tensor(np.asanyarray(img.dataobj))
		shape = img.shape
		img = interpolate(
			img.reshape(1,1,shape[0],shape[1],shape[2]),
			size=(128,128,128),
			mode='trilinear'
		)
		img = torch.clamp(img, -1000,1000)
		img = img/1000
		if ims is not None:
			ims = torch.concat((ims,img))
		else:
			ims = img

		return ims

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', type=str, default='')
	params = parser.parse_args()

	dirs = sorted([f for f in os.listdir(params.path) if os.path.isdir(f)],
		key=lambda x: int(x))

	for d in dirs:
		subdirs = sorted([f for f in os.listdir(d) if os.path.isdir(f)],
				key=lambda x: int(x))
		for sub in subdirs:
			files = sorted([f for f in os.listdir(os.path.join(d,sub)) if f.startswith('cbct')],
							key=lambda x: int(x[4:5]) if x[4:6].endswith('_') else int(x[4:6]))
			if len(files)<2:
				print(d, sub)
				print(len(files))

	ims = process(files[-1])
	print(ims.shape)


if __name__ == '__main__':
	main()
