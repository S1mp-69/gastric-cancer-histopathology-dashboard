import os
import numpy as np
from glob import glob
from PIL import Image
from stain_san.StainSAN import StainSAN
from argparse import ArgumentParser

parser = ArgumentParser(description='Stain SAN')
parser.add_argument('--train-input-dir', required=True, type=str)
parser.add_argument('--train-output-dir', required=True, type=str)
parser.add_argument('--test-dirs', nargs='+', required=True, type=str)
parser.add_argument('--extractor-type', required=True, type=str)
parser.add_argument('--n-jobs', required=True, type=int)
args = parser.parse_args()


train_input_dir = args.train_input_dir
train_output_dir = args.train_output_dir
test_dirs = args.test_dirs
extractor_type = args.extractor_type
n_jobs = args.n_jobs

os.makedirs(train_output_dir, exist_ok=True)

test_stain = []
for test_dir in test_dirs:
    test_adaptor = StainSAN(test_dir, None)
    test_adaptor.extract_stain(extractor_type, n_jobs)
    test_stain += test_adaptor.stain

train_adaptor = StainSAN(train_input_dir, train_output_dir)
train_adaptor.extract_stain(extractor_type, 50)
train_adaptor.mix_images(target_stain=test_stain, delta_H=0.2, n_jobs=50)
