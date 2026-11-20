import os
import numpy as np
import spams

from glob import glob
from PIL import Image
from joblib import dump, load, Parallel, delayed
from tqdm import tqdm

from stain_san.utils import *


def get_stain_svd(img, Io=255, alpha=1, beta1=0.3, beta2=np.Inf):
    # check if image is blank
    if np.std(img) == 0.0:
        return None, img[0, 0, 0], 0

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    od = rgb2od(img, Io)

    # remove too bright or too dark pixels
    od_hat = od[(np.sum(od**2, axis=1) > beta1**2) & (np.sum(od**2, axis=1) < beta2**2)]
    _, eig_vecs = np.linalg.eigh(np.cov(od_hat.T))
    coord_vecs = eig_vecs[:, 1:3]

    # project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
    coord_hat = od_hat @ coord_vecs
    phi = np.arctan2(coord_hat[:, 1], coord_hat[:, 0])
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)
    stain1 = coord_vecs @ np.array([(np.cos(min_phi), np.sin(min_phi))]).T
    stain2 = coord_vecs @ np.array([(np.cos(max_phi), np.sin(max_phi))]).T

    # make one vector corresponding to the first color and the other vector corresponding to the second color
    if stain1[0] > stain2[0]:
        stain = np.array((stain1[:, 0], stain2[:, 0])).T
    else:
        stain = np.array((stain2[:, 0], stain1[:, 0])).T
    W = stain
    W = np.clip(W, 0, None)
    H = np.linalg.lstsq(W, od.T, rcond=None)[0]
    H = np.clip(H, 0, None)
    residual = od.T - (W @ H)

    return W, H, residual


def get_stain_nmf(img, Io=255, lamb=0.1, beta1=0.3, beta2=np.Inf):
    # TODO: Try sklearn.decomposition.DictionaryLearning
    # check if image is blank
    if np.std(img) == 0.0:
        return None, img[0, 0, 0], 0

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    od = rgb2od(img, Io)

    # remove too bright or too dark pixels
    od_hat = od[(np.sum(od**2, axis=1) > beta1**2) & (np.sum(od**2, axis=1) < beta2**2)]

    # compute W
    W = spams.trainDL(od_hat.T, K=2, lambda1=lamb, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False)
    if W[0, 0] < W[0, 1]:
        W = W[:, [1, 0]]
    H = np.linalg.lstsq(W, od.T, rcond=None)[0]
    residual = od.T - (W @ H)

    return W, H, residual


def get_stain(file, extractor_type, beta1, beta2):
    file_name = os.path.basename(file)
    img = np.array(Image.open(file))
    size = img.shape
    if extractor_type == 'nmf':
        W, H, _ = get_stain_nmf(img, beta1=beta1, beta2=beta2)
    elif extractor_type == 'svd':
        W, H, _ = get_stain_svd(img, beta1=beta1, beta2=beta2)

    if len(H.shape) != 2 or H.shape[0] != 2:
        W, H99 = None, None
    else:
        H99 = np.percentile(H, 99, axis=1).reshape((2, 1))

    return (file_name, size, W, H99)

def get_od_from_file(img_file):
    img = np.array(Image.open(img_file))
    img = img.reshape((-1, 3))
    od = rgb2od(img)

    return od


def gaussian_update(prev, gamma, mu, sigma):
    # gaussian metropolis-hastings update
    # TODO: implement other distributions
    cand = prev + np.random.normal(0.0, gamma, prev.shape)
    p = np.min([1.0, np.exp(np.sum((prev - mu)**2 - (cand - mu)**2) / sigma**2)])
    u = np.random.uniform(0, 1)
    return cand if u < p else prev


class StainSAN(object):
    """
    Proposed model
    """

    def __init__(self, input_dir, output_dir, beta1=0.3, beta2=np.Inf):
        # if input_dir is None:
        #     input_dir = os.getcwd()
        # if output_dir is None:
        #     output_dir = os.path.join(input_dir, 'output')
        self.input_dir = input_dir
        self.output_dir = output_dir
        # if len(size) == 1:
        #     size = [size, size, 3]
        # elif len(size) == 2:
        #     size = list(size) + [3]
        # self.size = size
        self.stain = None
        self.beta1 = beta1  # luminosity threshold
        self.beta2 = beta2  # luminosity threshold

    def extract_stain(self, extractor_type, n_jobs):
        stain_path = os.path.join(self.input_dir, extractor_type, 'stain.joblib')
        if os.path.exists(stain_path):
            self.stain = load(stain_path)
        else:
            files = glob(os.path.join(self.input_dir, '*.jpg'))
            files += glob(os.path.join(self.input_dir, '*.png'))
            files += glob(os.path.join(self.input_dir, '*.tif'))

            self.stain = Parallel(n_jobs=n_jobs)(delayed(get_stain)(file, extractor_type, self.beta1, self.beta2)
                                                 for file in tqdm(files))
            os.makedirs(os.path.join(self.input_dir, extractor_type), exist_ok=True)
            dump(self.stain, stain_path)

    def normalize_images(self, W_0, H99_0, n_jobs):
        def normalize_image(file_name, size, W, H99):
            img_file = os.path.join(self.input_dir, file_name)
            od = get_od_from_file(img_file)
            if W is None or H99 is None:
                image = np.array(Image.open(img_file))
                Image.fromarray(image).save(os.path.join(self.output_dir, file_name))
                return
            H = np.linalg.lstsq(W, od.T, rcond=None)[0]
            # H99 = np.percentile(H, 99, axis=1).reshape((2, 1))
            scale = H99_0 / H99
            H = scale * H
            W = W_0
            adapted_image = reconstruct_image(W, H, od, self.beta1, self.beta2, size)
            Image.fromarray(adapted_image).save(os.path.join(self.output_dir, file_name))

        Parallel(n_jobs=n_jobs)(delayed(normalize_image)(file_name, size, W, H99)
                                for file_name, size, W, H99 in tqdm(self.stain))

    # def augment_images(self, gamma_W, gamma_H, n_jobs):
    #     def augment_image(file_name, size, W, H99):
    #     # for file_name, size, W, H99 in tqdm(self.stain):
    #         img_file = os.path.join(self.input_dir, file_name)
    #         img = np.array(Image.open(img_file))
    #         img = img.reshape((-1, 3))
    #         od = rgb2od(img)
    #         H = np.linalg.lstsq(W, od.T, rcond=None)[0]
    #         H *= (1 + np.random.normal(0, gamma_H))  # since H99 is being augmented
    #         H = np.clip(H, 0, None)
    #         W += np.random.normal(0, gamma_W)
    #         W = np.clip(W, 0, None)
    #         adapted_image = reconstruct_image(W, H, od, self.beta1, self.beta2, size)
    #         Image.fromarray(adapted_image).save(os.path.join(self.output_dir, file_name))
    #
    #     Parallel(n_jobs=n_jobs)(delayed(augment_image)(file_name, size, W, H99)
    #                             for file_name, size, W, H99 in tqdm(self.stain))

    def augment_images(self, W_0, alpha_H, beta_H, n_jobs):
        def augment_image(file_name, size):
        # for file_name, size, W, H99 in tqdm(self.stain):
            img_file = os.path.join(self.input_dir, file_name)
            od = get_od_from_file(img_file)
            if W is None or H99 is None:
                image = np.array(Image.open(img_file))
                Image.fromarray(image).save(os.path.join(self.output_dir, file_name))
                return
            if W_0.shape[1] < 3:
                H = np.linalg.lstsq(W_0, od.T, rcond=None)[0]
            else:
                H = np.linalg.inv(W_0) @ od.T
            H = np.diag((1 + np.random.uniform(-alpha_H, alpha_H, size=H.shape[0]))) @ H
            H = (H.T + np.random.uniform(-beta_H, beta_H, size=H.shape[0])).T
            # H = np.clip(H, 0, None)
            adapted_image = reconstruct_image(W_0, H, od, self.beta1, self.beta2, size)
            Image.fromarray(adapted_image).save(os.path.join(self.output_dir, file_name))

        Parallel(n_jobs=n_jobs)(delayed(augment_image)(file_name, size)
                                for file_name, size, _, _ in tqdm(self.stain))

    def mix_images(self, target_stain, delta_H, n_jobs):
        def mix_image(file_name, size, W):
            img_file = os.path.join(self.input_dir, file_name)
            od = get_od_from_file(img_file)
            if W is None or H99 is None:
                image = np.array(Image.open(img_file))
                Image.fromarray(image).save(os.path.join(self.output_dir, file_name))
                return
            H = np.linalg.lstsq(W, od.T, rcond=None)[0]
            # H = np.diag((1 + np.random.uniform(-delta_H, delta_H, size=H.shape[0]))) @ H
            H *= 1 + np.random.uniform(-delta_H, delta_H)
            sample_idx = np.random.choice(len(target_stain))
            _, _, W_target, _ = target_stain[sample_idx]
            p = np.random.uniform(0, 1)
            W = (1 - p) * W + p * W_target
            W = np.clip(W, 0, None)
            adapted_image = reconstruct_image(W, H, od, self.beta1, self.beta2, size)
            Image.fromarray(adapted_image).save(os.path.join(self.output_dir, file_name))

        Parallel(n_jobs=n_jobs)(delayed(mix_image)(file_name, size, W)
                                for file_name, size, W, _ in tqdm(self.stain))

    def update_images(self, gamma_W, mu_W, sigma_W, delta_H, n_steps, n_jobs):
        def update_image(file_name, size, W):
            img_file = os.path.join(self.input_dir, file_name)
            od = get_od_from_file(img_file)
            if W is None or H99 is None:
                image = np.array(Image.open(img_file))
                Image.fromarray(image).save(os.path.join(self.output_dir, file_name))
                return
            H = np.linalg.lstsq(W, od.T, rcond=None)[0]
            H = np.diag((1 + np.random.uniform(-delta_H, delta_H, size=H.shape[0]))) @ H
            for _ in range(n_steps):
                W = gaussian_update(W, gamma_W, mu_W, sigma_W)
                W = np.clip(W, 0, None)
            adapted_image = reconstruct_image(W, H, od, self.beta1, self.beta2, size)
            Image.fromarray(adapted_image).save(os.path.join(self.output_dir, file_name))

        Parallel(n_jobs=n_jobs)(delayed(update_image)(file_name, size, W)
                                for file_name, size, W, _ in tqdm(self.stain))

    # def san_images(self, mu_W, sigma_W, delta_H, n_jobs):
    #     def san_image(file_name, size, W, H99):
    #         img_file = os.path.join(self.input_dir, file_name)
    #         od = get_od_from_file(img_file)
    #         H = np.linalg.lstsq(W, od.T, rcond=None)[0]
    #         H = np.diag((1 + np.random.uniform(-delta_H, delta_H, size=H.shape[0]))) @ H
    #         W = mu_W + np.random.normal(0, sigma_W, size=W.shape)
    #         W = np.clip(W, 0, None)
    #         adapted_image = reconstruct_image(W, H, od, self.beta1, self.beta2, size)
    #         Image.fromarray(adapted_image).save(os.path.join(self.output_dir, file_name))
    #
    #     Parallel(n_jobs=n_jobs)(delayed(san_image)(file_name, size, W, H99)
    #                             for file_name, size, W, H99 in tqdm(self.stain))

    def san_images(self, mu_W, sigma_W, mu_H99=None, sigma_H99=None, delta_H=None, n_jobs=-1):
        sigma_H99 = np.array(sigma_H99)
        sigma_W = np.array(sigma_W)
        def san_image(file_name, size, W, H99):
            img_file = os.path.join(self.input_dir, file_name)
            od = get_od_from_file(img_file)
            if W is None or H99 is None:
                image = np.array(Image.open(img_file))
                Image.fromarray(image).save(os.path.join(self.output_dir, file_name))
                return
            H = np.linalg.lstsq(W, od.T, rcond=None)[0]
            if delta_H is None:
                H99 = np.percentile(H, 99, axis=1).reshape((2, 1))
                if sigma_H99.shape == () or sigma_H99.shape[0] == 1:
                    san_H99 = mu_H99 + np.random.normal(0, sigma_H99, size=H99.shape)
                else:
                    san_H99 = mu_H99 + np.random.normal(0, sigma_H99)
                san_H99 = np.clip(san_H99, 0, None)
                scale = san_H99 / H99
                H = scale * H
            else:
                # H = np.diag((1 + np.random.uniform(-delta_H, delta_H, size=H.shape[0]))) @ H # a la stain augmentation
                H *= 1 + np.random.uniform(-delta_H, delta_H) # a la stain mix-up
            if sigma_W.shape == () or sigma_W.shape[0] == 1:
                W = mu_W + np.random.normal(0, sigma_W, size=W.shape)
            else:
                W = mu_W + np.random.normal(0, sigma_W)
            W = np.clip(W, 0, None)
            adapted_image = reconstruct_image(W, H, od, self.beta1, self.beta2, size)
            Image.fromarray(adapted_image).save(os.path.join(self.output_dir, file_name))

        Parallel(n_jobs=n_jobs)(delayed(san_image)(file_name, size, W, H99)
                                for file_name, size, W, H99 in tqdm(self.stain))
