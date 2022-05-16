import numpy as np
import os
import nibabel as nib
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', type=str, default='')
	params = parser.parse_args()
	files = sorted([f for f in os.listdir(params.path) if f.startswith('cbct')],
		key=lambda x: x[4:5] if x[4:6].endswith('_') else x[4:6])
	print(files)
	for f in files:
		img = nib.load(os.path.join(params.path,f))
		#print(img.header)


if __name__ == '__main__':
	main()
