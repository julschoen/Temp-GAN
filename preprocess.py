import numpy as np
import os
import nibabel as nib
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', type=str, default='')
	params = parser.parse_args()
	files = sorted([f for f in os.listdir(params.path) if f.startswith('cbct')], key=lambda x: x[:6])
	print(files)
	for f in files:
		print(f[4:6])
		img = nib.load(os.path.join(params.path,f))
		#print(img.header)


if __name__ == '__main__':
	main()
