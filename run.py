import os
import scipy.misc
import numpy as np
import argparse

from consensus_gan.train import train
from consensus_gan.inputs import (
    get_filename_queue,
    get_input_image, get_input_cifar10,
    create_batch
)
from consensus_gan.utils import pp
from consensus_gan import models
import tensorflow as tf

parser = argparse.ArgumentParser(description='Train and run a tVAE.')
# Architecture
parser.add_argument('--image-size', default=128, type=int, help='Size of image crops.')
parser.add_argument('--output-size', default=64, type=int, help='Size of samples.')
parser.add_argument('--c-dim', default=3, type=int, help='Number of channels.')
parser.add_argument('--z-dim', default=512, type=int, help='Dimensionality of the latent space.')
parser.add_argument('--gf-dim', default=64, type=int, help='Number of filters to use for generator.')
parser.add_argument('--df-dim', default=64, type=int, help='Number of filters to use for discriminator.')
parser.add_argument('--reg-param', default=10., type=float, help='Regularization parameter.')
parser.add_argument('--g-architecture', default='conv0', type=str, help='Architecture for generator.')
parser.add_argument('--d-architecture', default='conv0', type=str, help='Architecture for discriminator.')
parser.add_argument('--gan-type', default='standard', type=str, help='Which type of GAN to use.')

# Training
parser.add_argument('--optimizer', default='conopt', type=str, help='Which optimizer to use.')
parser.add_argument('--altgd-gsteps', default='1', type=int, help='How many training steps to use for generator.')
parser.add_argument('--altgd-dsteps', default='1', type=int, help='How many training steps to use for discriminator.')
parser.add_argument('--nsteps', default=200000, type=int, help='Number of steps to run training.')
parser.add_argument('--ntest', default=500, type=int, help='How often to run tests.')
parser.add_argument('--learning-rate', default=1e-4, type=float, help='Learning rate for the model.')
parser.add_argument('--batch-size', default=64, type=int, help='Batchsize for training.')
parser.add_argument('--log-dir', default='./logs', type=str, help='Where to store log and checkpoint files.')
parser.add_argument('--sample-dir', default='./samples', type=str, help='Where to put samples during training.')
parser.add_argument('--is-inception-scores', default=False, action='store_true',
                        help='Wether to compute inception scores.')
parser.add_argument('--inception-dir', default='./inception', type=str, help='Where to put inception network.')

parser.add_argument('--dataset', default='cifar-10', type=str, help='Which data set to use.')
parser.add_argument('--data-dir', default='./data', type=str, help='Where data data is stored..')
parser.add_argument('--split', default='train', type=str, help='Which split to use.')

def main():
    args = parser.parse_args()
    pp.pprint(vars(args))

    # Data
    filename_queue = get_filename_queue(
        split_file=os.path.join(args.data_dir, 'splits', args.dataset, args.split + '.lst'),
        data_dir=os.path.join(args.data_dir, args.dataset)
    )

    if args.dataset == "cifar-10":
        image, label = get_input_cifar10(filename_queue)
        output_size = 32
        c_dim = 3
    else:
        image = get_input_image(filename_queue,
            output_size=args.output_size, image_size=args.image_size, c_dim=args.c_dim
        )
        output_size = args.output_size
        c_dim = args.c_dim

    image_batch = create_batch([image], batch_size=args.batch_size,
        num_preprocess_threads=16, min_queue_examples=10000)

    config = vars(args)

    generator = models.get_generator(args.g_architecture,
        output_size=args.output_size, c_dim=args.c_dim, f_dim=args.gf_dim)

    discriminator = models.get_discriminator(args.d_architecture,
        output_size=args.output_size, c_dim=args.c_dim, f_dim=args.df_dim)

    train(generator, discriminator, image_batch, config)


if __name__ == '__main__':
    main()
