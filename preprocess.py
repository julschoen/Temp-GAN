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
			size=(128,128,128),
			mode='trilinear'
		)
		img = torch.clamp(img, -1000,1000)
		img = img/1000
		if ims is not None:
			ims = torch.concat((ims,img))
		else:
			ims = img

	return ims.numpy()

def combine(dirs, params):
	data = None
	for d in dirs:
		if d in choice:
			continue
		files = sorted([f for f in os.listdir(os.path.join(params.save_path, d)) if f.endswith('.npz')],
						key=lambda x: int(x))
		pat = None
		for f in files:
			ims = np.load(os.path.join(params.save_path, d, f))['x']
			if pat is None:
				pat = ims.reshape(1,-1,1,128,128,128)
			else:
				pat = np.concatenate((pat, ims.reshape(1,-1,1,128,128,128)))

		if data is None:
			data = pat.reshape(1,pat.shape[0],-1,1,128,128,128)
		else:
			data = np.concatenate((data, pat.reshape(1,pat.shape[0],-1,1,128,128,128)))
	return data

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_path', type=str, default='')
	parser.add_argument('-s', '--save_path', type=str, default='data')
	params = parser.parse_args()

	"""
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
				os.makedirs(os.path.join(params.save_path, d), exist_ok=True)
				ims = process(os.path.join(params.data_path, d, sub), files)
				np.savez_compressed(os.path.join(params.save_path, d,f'{sub}.npz'), x=ims)
				print(f'Patient {d}, Series {sub}, Number of Scans {ims.shape[0]}')
	"""
	dirs = sorted([f for f in os.listdir(params.save_path) if os.path.isdir(os.path.join(params.save_path, f))],
		key=lambda x: int(x))

	choice = np.random.choice(dirs, size=int(len(dirs)*0.1))
	np.savez_compressed(os.path.join(params.save_path, 'test_pat.npz'), x=np.array(choice))
	
	train = []
	test = choice

	for d in dirs:
		if d in choice:
			continue
		else:
			train.append(d)

	train_data = combine(train, params)
	print(train.shape)
	np.savez_compressed(os.path.join(params.save_path, 'train.npz'), x=train_data)
	test_data = combine(choice, params)
	print(test_data.shape)
	np.savez_compressed(os.path.join(params.save_path, 'test.npz'), x=test_data)





if __name__ == '__main__':
	main()
