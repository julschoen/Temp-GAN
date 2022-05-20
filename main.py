from temp_data_handler import DATA
from trainer import Trainer
import argparse


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--niters', type=int, default=5000, help='Number of training iterations')
	parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
	parser.add_argument('--z_size', type=int, default=512, help='Latent space dimension')
	parser.add_argument('--filterG', type=int, default=64, help='Number of filters G')
	parser.add_argument('--filterD', type=int, default=64, help='Number of filters D')
	parser.add_argument('--iterD', type=int, default=1, help='Number of D iters per iter')
	parser.add_argument('--lrG', type=float, default=1e-4, help='Learning rate G')
	parser.add_argument('--lrD', type=float, default=1e-4, help='Learning rate D')
	parser.add_argument('--data_path', type=str, default='data',help='Path to data.')
	parser.add_argument('--ngpu', type=int, default=2, help='Number of GPUs')
	parser.add_argument('--steps_per_log', type=int, default=10, help='Output Iterations')
	parser.add_argument('--steps_per_img_log', type=int, default=50, help='Image Save Iterations')
	parser.add_argument('--log_dir', type=str, default='log', help='Save Location')
	parser.add_argument('--device', type=str, default='cuda', help='Torch Device Choice')
	parser.add_argument('--att', type=bool, default=True, help='Use Attention in BigGAN')
	params = parser.parse_args()
	print(params)
	dataset_train = DATA(path=params.data_path)

	trainer = Trainer(dataset_train, params=params)
	trainer.train()

if __name__ == '__main__':
	main()
