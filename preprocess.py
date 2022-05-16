import numpy as np
import os
import nibabel as nib
import argparse

def main(params):
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', type=str, default='')
	params = parser.parse_args()
	img = nib.load(params.path)
	print(img.header)


if __name__ == '__main__':
	main()
