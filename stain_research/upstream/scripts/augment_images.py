import os
import numpy as np
from glob import glob
from PIL import Image
from stain_san.StainSAN import StainSAN
from argparse import ArgumentParser

parser = ArgumentParser(description='Stain SAN')
parser.add_argument('--input-dir', required=True, type=str)
parser.add_argument('--output-dir', required=True, type=str)
parser.add_argument('--extractor-type', required=True, type=str)
parser.add_argument('--n-jobs', required=True, type=int)
args = parser.parse_args()


input_dir = args.input_dir
output_dir = args.output_dir
extractor_type = args.extractor_type
n_jobs = args.n_jobs

os.makedirs(output_dir, exist_ok=True)


# def get_image_size(dir):
#     image_files = glob(os.path.join(dir, '*'))
#     image = np.array(Image.open(image_files[0]))
#     size = image.shape
#
#     return size


# train_size = get_image_size(train_input_dir)
# test_size = get_image_size(test_input_dir)

adaptor = StainSAN(input_dir, output_dir)
# TODO: get rid of this step
adaptor.extract_stain(extractor_type, 50)

# W_0 = np.array([[0.65, 0.07],
#                 [0.70, 0.99],
#                 [0.29, 0.11]])

W_0 = np.array([[0.65, 0.07, 0.27],
                [0.70, 0.99, 0.57],
                [0.29, 0.11, 0.78]])

# W_0 = np.array([[1, 0, 0],
#                 [0, 1, 0],
#                 [0, 0, 1]])

adaptor.augment_images(W_0=W_0, alpha_H=0.2, beta_H=0.2, n_jobs=n_jobs)
