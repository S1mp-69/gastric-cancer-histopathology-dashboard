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
args = parser.parse_args()


train_input_dir = args.train_input_dir
train_output_dir = args.train_output_dir
test_input_dir = args.test_input_dir
test_output_dir = args.test_output_dir
extractor_type = args.extractor_type
n_jobs = args.n_jobs
test_only = args.test_only

os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

train_adaptor = StainSAN(train_input_dir, train_output_dir)
train_adaptor.extract_stain(extractor_type, n_jobs)

W_0 = np.median([W for file_name, size, W, H99 in train_adaptor.stain], axis=0)
sigma_W = np.sqrt(np.sum(np.var([W for file_name, size, W, H99 in train_adaptor.stain], axis=0)) / 6)
gamma_W = sigma_W / 2

if not test_only:
    train_adaptor.update_images(gamma_W=gamma_W, mu_W=W_0, sigma_W=sigma_W, delta_H=0.2, n_steps=8, n_jobs=50)
test_adaptor = StainSAN(test_input_dir, test_output_dir)
test_adaptor.extract_stain(extractor_type, n_jobs)
test_adaptor.update_images(gamma_W=gamma_W, mu_W=W_0, sigma_W=sigma_W, delta_H=0.2, n_steps=8, n_jobs=50)
