import os
import numpy as np
from glob import glob
from PIL import Image
from joblib import dump, load, Parallel, delayed
from tqdm import tqdm
from stain_san import get_stain
from argparse import ArgumentParser


parser = ArgumentParser(description='Extract stain matrices')
parser.add_argument('--input-dir', '-i', required=True, type=str)
parser.add_argument('--stain-dir', '-s', type=str)
parser.add_argument('--extractor-type', required=True, type=str)
parser.add_argument('--n-jobs', type=int)
args = parser.parse_args()


input_dir = args.input_dir
stain_dir = args.stain_dir
extractor_type = args.extractor_type
if stain_dir is None:
    stain_dir = os.path.join(input_dir, extractor_type)
os.makedirs(stain_dir, exist_ok=True)
n_jobs = args.n_jobs

files = glob(os.path.join(input_dir, '*.jpg'))
files += glob(os.path.join(input_dir, '*.png'))
files += glob(os.path.join(input_dir, '*.tif'))

# def get_stain(file, extractor_type):
#     file_name = os.path.basename(file)
#     img = np.array(Image.open(file))
#     if extractor_type == 'nmf':
#         W, H, _ = get_stain_nmf(img, 255, 0.1)
#     elif extractor_type == 'svd':
#         W, H, _ = get_stain_svd(img, 255)
#
#     H99 = np.percentile(H, 99, axis=1).reshape((2, 1))
#
#     return (file_name, W, H99)

stain = Parallel(n_jobs=n_jobs)(delayed(get_stain)(file, extractor_type) for file in tqdm(files))
dump(stain, os.path.join(stain_dir, 'stain.joblib'))
