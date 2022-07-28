import numpy as np
import os
import nibabel as nib
import argparse
import torch
from torch.nn.functional import interpolate
import re

def process(path, patient, phases):
	ims = None
	phases = np.sort(phases)
	for phase in phases:
		img = nib.load(os.path.join(path,f'patient{patient}phase{phase}.0.nii.gz'))
		if not np.allclose(img.header['dim'], [3,512,512,145,1,1,1,1]):
			print(patient)
			print(img.header['dim'])
		if not np.allclose(img.header['pixdim'],  [1., 0.9765625, 0.9765625, 2., 1., 1., 1., 1.]):
			print(patient)
			print(img.header['pixdim'])
		img = torch.Tensor(np.asanyarray(img.dataobj))
		shape = img.shape
		img = interpolate(
			img.reshape(1,1,shape[0],shape[1],shape[2]),
			size=(128,128,64),
			mode='trilinear'
		)
		img = torch.clamp(img, -1000,500)
		img = img+250
		img = img/750
		img = torch.clamp(img, -1,1)
		if ims is not None:
			ims = torch.concat((ims,img))
		else:
			ims = img
		break

	return ims.numpy()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_path', type=str, default='')
	parser.add_argument('-s', '--save_path', type=str, default='data')
	params = parser.parse_args()
	pat = {}
	for f in os.listdir(params.data_path):
		if not f.endswith('gz'):
			continue
		i = re.findall(r'\d+', f)
		if int(i[0]) in pat.keys():
			pat[int(i[0])].append(int(i[1]))
		else:
			pat[int(i[0])] = [int(i[1])]
	print(f'Preprocessing {len(pat.keys())} patients')
	os.makedirs(params.save_path, exist_ok=True)
	for p in pat.keys():
		phase = pat[p]
		ims = process(params.data_path, p, phase)
	"""
		if ims.shape[0] > 9:
			np.savez_compressed(os.path.join(params.save_path,f'{p}.npz'), x=ims)
			print(f'Patient {p}, Number of Scans {ims.shape[0]}')
	pat = [f for f in os.listdir(params.save_path) if f.endswith('npz')]
	print(f'Done Processing {len(pat)} patients')
	test = np.random.choice(pat, size=int(len(pat)*0.1))
	train = []
	for d in pat:
		if d in test:
			continue
		else:
			train.append(d)

	np.savez_compressed(os.path.join(params.save_path, 'test_pat.npz'), x=np.array(test))
	np.savez_compressed(os.path.join(params.save_path, 'train_pat.npz'), x=np.array(train))
	"""


if __name__ == '__main__':
	main()
