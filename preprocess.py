import numpy as np
import os
import nibabel as nib
import argparse
import torch
from torch.nn.functional import interpolate

def process(path, files):
	ims = None
	for f in files:
		if not f.endswith('.gz'):
			continue
		img = nib.load(os.path.join(path,f))
		img = torch.Tensor(np.asanyarray(img.dataobj))
		shape = img.shape
		img = interpolate(
			img.reshape(1,1,shape[0],shape[1],shape[2]),
			size=(128,128,64),
			mode='trilinear'
		)
		img = torch.clamp(img, -1000,1000)
		img = img/1000
		if ims is not None:
			ims = torch.concat((ims,img))
		else:
			ims = img

	return ims.numpy()

def clean(path, dirs):
	new_dirs = np.copy(dirs)
	for d in dirs:
		files = sorted([f for f in os.listdir(os.path.join(path,d)) if f.endswith('npz')],
							key=lambda x: int(x[:-4]))
		if len(files) > 1:
			for i, f in enumerate(files):
				if i == 0:
					continue
				else:
					new_path = os.path.join(path, d+f'_{i}')
					os.makedirs(new_path, exist_ok=True)
					os.replace(os.path.join(path, d, f), os.path.join(new_path, f))
					new_dirs = np.append(new_dirs, d+f'_{i}')

	return new_dirs

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_path', type=str, default='')
	parser.add_argument('-s', '--save_path', type=str, default='data')
	params = parser.parse_args()

	dirs = sorted([f for f in os.listdir(params.data_path) if os.path.isdir(os.path.join(params.data_path, f))],
		key=lambda x: int(x))
	os.makedirs(params.save_path, exist_ok=True)
	for d in dirs:
		subdirs = sorted([f for f in os.listdir(os.path.join(params.data_path, d)) 
									if os.path.isdir(os.path.join(params.data_path,d,f))],
				key=lambda x: int(x))
		for sub in subdirs:
			files = sorted([f for f in os.listdir(os.path.join(params.data_path,d,sub)) if f.startswith('cbct')],
							key=lambda x: int(x[4:5]) if x[4:6].endswith('_') else int(x[4:6]))
			if len(files)>1:
				ims = process(os.path.join(params.data_path, d, sub), files)
				if ims.shape[0] > 2:
					os.makedirs(os.path.join(params.save_path, d), exist_ok=True)
					np.savez_compressed(os.path.join(params.save_path, d,f'{sub}.npz'), x=ims)
					print(f'Patient {d}, Series {sub}, Number of Scans {ims.shape[0]}')
	
	dirs = sorted([f for f in os.listdir(params.save_path) if os.path.isdir(os.path.join(params.save_path, f))],
		key=lambda x: int(x))

	test = np.random.choice(dirs, size=int(len(dirs)*0.1))
	train = []
	for d in dirs:
		if d in test:
			continue
		else:
			train.append(d)

	train = clean(params.save_path, train)
	test = clean(params.save_path, test)

	np.savez_compressed(os.path.join(params.save_path, 'test_pat.npz'), x=np.array(test))
	np.savez_compressed(os.path.join(params.save_path, 'train_pat.npz'), x=np.array(train))




if __name__ == '__main__':
	main()
