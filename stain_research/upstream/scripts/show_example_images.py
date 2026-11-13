import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from os.path import join, dirname
from glob import glob
from PIL import Image
from argparse import ArgumentParser


parser = ArgumentParser(description='Show example images')
parser.add_argument('--original-dir', required=True, type=str)
parser.add_argument('--normalized-dir', required=True, type=str)
parser.add_argument('--augmented-dir', required=True, type=str)
parser.add_argument('--mixed-dir', required=True, type=str)
parser.add_argument('--san-dir', required=True, type=str)
parser.add_argument('--output-path', required=True, type=str)
parser.add_argument('--n-images', required=True, type=int)
parser.add_argument('--add-count', type=int)
parser.add_argument('--center-size', type=int)
args = parser.parse_args()


# store positive and negative images
def store_image_files(image_dir, pos_subjs, neg_subjs, n_images):
    pos_files, neg_files = [], []
    i = add_count
    while len(pos_files) < n_images:
        pos_candidates = glob(join(image_dir, str(pos_subjs[i]) + '*.jpg')) + \
                         glob(join(image_dir, str(pos_subjs[i]) + '*.png'))
        pos_candidates.sort()
        if len(pos_candidates) > 0:
            pos_files.append(pos_candidates[0])
        i += 1

    i = add_count
    while len(neg_files) < n_images:
        neg_candidates = glob(join(image_dir, str(neg_subjs[i]) + '*.jpg')) + \
                         glob(join(image_dir, str(neg_subjs[i]) + '*.png'))
        neg_candidates.sort()
        if len(neg_candidates) > 0:
            neg_files.append(neg_candidates[0])
        i += 1

    return pos_files, neg_files


# get center zoomed in images
def get_center_image(image, size):
    if size is None:
        return image
    h, w = image.shape[0], image.shape[1]
    center_image = image[(h//2 - size//2):(h//2 + size//2), (w//2 - size//2):(w//2 + size//2)]
    return center_image


# draw plots of positive and negative images
def plot_pos_neg_images(pos_files, neg_files, n_row, i, axes, size):
    pos_image = np.array(Image.open(pos_files[i]))
    pos_image = get_center_image(pos_image, size)
    ax = axes[n_row, i]
    ax.axis('off')
    ax.imshow(pos_image)
    neg_image = np.array(Image.open(neg_files[i]))
    neg_image = get_center_image(neg_image, size)
    ax = axes[n_row, i + n_images]
    ax.axis('off')
    ax.imshow(neg_image)


original_dir = args.original_dir
normalized_dir = args.normalized_dir
augmented_dir = args.augmented_dir
mixed_dir = args.mixed_dir
san_dir = args.san_dir
output_path = args.output_path
os.makedirs(dirname(output_path), exist_ok=True)
n_images = args.n_images
add_count = args.add_count
center_size = args.center_size

# collect subject ids grouped by labels
labels_path = join(dirname(dirname(original_dir)), 'mil/he/labels.csv')
labels = pd.read_csv(labels_path, index_col=0)
labels = labels['er']
pos_subjs = labels[labels == 1].index
neg_subjs = labels[labels == 0].index

# store files
original_pos_files, original_neg_files = store_image_files(original_dir, pos_subjs, neg_subjs, n_images)
normalized_pos_files, normalized_neg_files = store_image_files(normalized_dir, pos_subjs, neg_subjs, n_images)
augmented_pos_files, augmented_neg_files = store_image_files(augmented_dir, pos_subjs, neg_subjs, n_images)
mixed_pos_files, mixed_neg_files = store_image_files(mixed_dir, pos_subjs, neg_subjs, n_images)
san_pos_files, san_neg_files = store_image_files(san_dir, pos_subjs, neg_subjs, n_images)

# draw plots
fig, axes = plt.subplots(figsize=(10*n_images, 5*5), ncols=2 * n_images, nrows=5)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.05, hspace=0.05)
for i in range(n_images):
    plot_pos_neg_images(original_pos_files, original_neg_files, 0, i, axes, center_size)
    plot_pos_neg_images(normalized_pos_files, normalized_neg_files, 1, i, axes, center_size)
    plot_pos_neg_images(augmented_pos_files, augmented_neg_files, 2, i, axes, center_size)
    plot_pos_neg_images(mixed_pos_files, mixed_neg_files, 3, i, axes, center_size)
    plot_pos_neg_images(san_pos_files, san_neg_files, 4, i, axes, center_size)

    # original_pos_image = Image.open(original_pos_files[i])
    # ax = axes[0, i]
    # ax.axis('off')
    # ax.imshow(original_pos_image)
    # original_neg_image = Image.open(original_neg_files[i])
    # ax = axes[0, i + n_images]
    # ax.axis('off')
    # ax.imshow(original_neg_image)
    #
    # normalized_pos_image = Image.open(normalized_pos_files[i])
    # ax = axes[1, i]
    # ax.axis('off')
    # ax.imshow(normalized_pos_image)
    # normalized_neg_image = Image.open(normalized_neg_files[i])
    # ax = axes[1, i + n_images]
    # ax.axis('off')
    # ax.imshow(normalized_neg_image)
    #
    # augmented_pos_image = Image.open(augmented_pos_files[i])
    # ax = axes[2, i]
    # ax.axis('off')
    # ax.imshow(augmented_pos_image)
    # augmented_neg_image = Image.open(augmented_neg_files[i])
    # ax = axes[2, i + n_images]
    # ax.axis('off')
    # ax.imshow(augmented_neg_image)
    #
    # mixed_pos_image = Image.open(mixed_pos_files[i])
    # ax = axes[3, i]
    # ax.axis('off')
    # ax.imshow(mixed_pos_image)
    # mixed_neg_image = Image.open(mixed_neg_files[i])
    # ax = axes[3, i + n_images]
    # ax.axis('off')
    # ax.imshow(mixed_neg_image)
    #
    # san_pos_image = Image.open(san_pos_files[i])
    # ax = axes[4, i]
    # ax.axis('off')
    # ax.imshow(san_pos_image)
    # san_neg_image = Image.open(san_neg_files[i])
    # ax = axes[4, i + n_images]
    # ax.axis('off')
    # ax.imshow(san_neg_image)

fig.savefig(output_path)





#
# original_files = glob(os.path.join(original_dir, '*.jpg'))
# original_files += glob(os.path.join(original_dir, '*.png'))
# original_files += glob(os.path.join(original_dir, '*.tif'))
# original_files.sort()
#
# normalized_files = glob(os.path.join(normalized_dir, '*.jpg'))
# normalized_files += glob(os.path.join(normalized_dir, '*.png'))
# normalized_files += glob(os.path.join(normalized_dir, '*.tif'))
# normalized_files.sort()
#
# augmented_files = glob(os.path.join(augmented_dir, '*.jpg'))
# augmented_files += glob(os.path.join(augmented_dir, '*.png'))
# augmented_files += glob(os.path.join(augmented_dir, '*.tif'))
# augmented_files.sort()
#
# mixed_files = glob(os.path.join(mixed_dir, '*.jpg'))
# mixed_files += glob(os.path.join(mixed_dir, '*.png'))
# mixed_files += glob(os.path.join(mixed_dir, '*.tif'))
# mixed_files.sort()
#
# san_files = glob(os.path.join(san_dir, '*.jpg'))
# san_files += glob(os.path.join(san_dir, '*.png'))
# san_files += glob(os.path.join(san_dir, '*.tif'))
# san_files.sort()
#
# fig, axes = plt.subplots(figsize=(20*n_images, 20*5), ncols=n_images, nrows=5)
# for i in range(n_images):
#     original_image = Image.open(original_files[i])
#     ax = axes[0, i]
#     ax.axis('off')
#     ax.imshow(original_image)
#
#     normalized_image = Image.open(normalized_files[i])
#     ax = axes[1, i]
#     ax.axis('off')
#     ax.imshow(normalized_image)
#
#     augmented_image = Image.open(augmented_files[i])
#     ax = axes[2, i]
#     ax.axis('off')
#     ax.imshow(augmented_image)
#
#     mixed_image = Image.open(mixed_files[i])
#     ax = axes[3, i]
#     ax.axis('off')
#     ax.imshow(mixed_image)
#
#     san_image = Image.open(san_files[i])
#     ax = axes[4, i]
#     ax.axis('off')
#     ax.imshow(san_image)
#
# fig.savefig(save_path)