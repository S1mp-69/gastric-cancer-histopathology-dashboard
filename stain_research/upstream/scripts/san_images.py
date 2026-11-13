import os
import numpy as np
from glob import glob
from PIL import Image
from stain_san.StainSAN import StainSAN
from argparse import ArgumentParser

parser = ArgumentParser(description='Stain SAN')
parser.add_argument('--train-input-dir', required=True, type=str)
parser.add_argument('--train-output-dir', required=True, type=str)
parser.add_argument('--test-input-dir', required=True, type=str)
parser.add_argument('--test-output-dir', required=True, type=str)
parser.add_argument('--extractor-type', required=True, type=str)
parser.add_argument('--n-jobs', required=True, type=int)
parser.add_argument('--test-only', action='store_true')
parser.add_argument('--case', required=True, type=str)
args = parser.parse_args()

np.random.seed(111)

train_input_dir = args.train_input_dir
train_output_dir = args.train_output_dir
test_input_dir = args.test_input_dir
test_output_dir = args.test_output_dir
extractor_type = args.extractor_type
n_jobs = args.n_jobs
test_only = args.test_only
case = args.case

os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

train_adaptor = StainSAN(train_input_dir, train_output_dir)
train_adaptor.extract_stain(extractor_type, n_jobs)

mu_W = np.median([W for file_name, size, W, H99 in train_adaptor.stain
                  if W is not None and H99 is not None], axis=0)

# Energy-preserving
sigma_W = np.sqrt(np.sum(np.var([W for file_name, size, W, H99 in train_adaptor.stain
                                 if W is not None and H99 is not None], axis=0)) / 6)

# Variance-preserving
# sigma_W = np.std([W for file_name, size, W, H99 in train_adaptor.stain], axis=0)

# Uniform
########## Check stain augmentation vs. stain mix-up!
if not test_only:
    if case == 'gaussian_one_zero_uniform_train_mixup': # default
        train_adaptor.san_images(mu_W=mu_W, sigma_W=sigma_W, delta_H=0.2, n_jobs=n_jobs)
    elif case == 'gaussian_half_zero_uniform_train_mixup':
        train_adaptor.san_images(mu_W=mu_W, sigma_W=sigma_W / 2, delta_H=0.2, n_jobs=n_jobs)
test_adaptor = StainSAN(test_input_dir, test_output_dir)
test_adaptor.extract_stain(extractor_type, n_jobs)
test_adaptor.san_images(mu_W=mu_W, sigma_W=0, delta_H=0, n_jobs=n_jobs)

# # Gaussian / Energy-preserving
# mu_H99 = np.median([H99 for file_name, size, W, H99 in train_adaptor.stain], axis=0)
# # sigma_H99 = np.sqrt(np.sum(np.var([H99 for file_name, size, W, H99 in train_adaptor.stain], axis=0)) / 2)
#
# # Gaussian / Variance-preserving
# sigma_H99 = np.std([H99 for file_name, size, W, H99 in train_adaptor.stain], axis=0)
#
#
# if not test_only:
#     train_adaptor.san_images(mu_W=mu_W, sigma_W=1 * sigma_W, mu_H99=mu_H99, sigma_H99=1 * sigma_H99, n_jobs=n_jobs)
# test_adaptor = StainSAN(test_input_dir, test_output_dir)
# test_adaptor.extract_stain(extractor_type, n_jobs)
# test_adaptor.san_images(mu_W=mu_W, sigma_W=0 * sigma_W, mu_H99=mu_H99, sigma_H99=0 * sigma_H99, n_jobs=n_jobs)
