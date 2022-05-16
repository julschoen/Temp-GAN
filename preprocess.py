import numpy as np
import os
import nibabel as nib
import argparse
import torch
from torch.nn.functional import interpolate

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', type=str, default='')
	params = parser.parse_args()
	files = sorted([f for f in os.listdir(params.path) if f.startswith('cbct')],
		key=lambda x: int(x[4:5]) if x[4:6].endswith('_') else int(x[4:6]))
	print(files)
	for f in files:
		img = nib.load(os.path.join(params.path,f))
		img_ = torch.Tensor(np.asanyarray(img.dataobj))
		img_ = interpolate(
			img_.reshape(1,1,384,384,64),
			size=(128,128,128),
			mode='trilinear'
		)
		img_ = torch.clamp(img_, -1000,1000)
		img_ = img_/1000
		print(img_.min(), img_.max())
		print(img_.shape)
		break


if __name__ == '__main__':
	main()
