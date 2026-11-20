import os
from abc import ABC, abstractmethod

import numpy as np
import spams
# from sklearn.decomposition import DictionaryLearning

from stain_san.utils import *


# class StainExtractor:
#     """
#     Stain extractor
#     """
#
#     def __init__(self, method='macenko', Io=255, alpha=0.5, lamb=0.1, beta1=0.1, beta2=np.Inf):
#         self.method = method
#         self.Io = Io
#         self.alpha = alpha
#         self.lamb = lamb
#         self.beta1 = beta1
#         self.beta2 = beta2
#
#     def get_stain_matrix(self, img):
#
#     @classmethod
#     def get_stain_macenko

class StainExtractor(ABC):
    """
    ABC stain extractor
    """

    @abstractmethod
    def get_stain_matrix(self, img):
        pass


class MacenkoStainExtractor(StainExtractor):
    """
    Macenko stain extractor
    """

    def __init__(self, Io=255, alpha=0.5, beta1=0.1, beta2=np.Inf):
        self.Io = Io
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2

    def get_stain_matrix(self, img):
        # check if image is blank
        if np.std(img) == 0.0:
            return None

        # reshape image
        img = img.reshape((-1, 3))

        # calculate optical density
        od = rgb2od(img, self.Io)

        # remove too bright or too dark pixels
        od_hat = od[(np.sum(od ** 2, axis=1) > self.beta1 ** 2) & (np.sum(od ** 2, axis=1) < self.beta2 ** 2)]

        _, eig_vecs = np.linalg.eigh(np.cov(od_hat.T))
        coord_vecs = eig_vecs[:, 1:3]

        # project on the plane spanned by the eigenvectors corresponding to the two largest eigenvalues
        coord_hat = od_hat.dot(coord_vecs)

        angle = np.arctan2(coord_hat[:, 1], coord_hat[:, 0])

        min_angle = np.percentile(angle, self.alpha)
        max_angle = np.percentile(angle, 100 - self.alpha)

        stain1 = coord_vecs.dot(np.array([(np.cos(min_angle), np.sin(min_angle))]).T)
        stain2 = coord_vecs.dot(np.array([(np.cos(max_angle), np.sin(max_angle))]).T)

        # make one vector corresponding to the first color and the other vector corresponding to the second color
        if stain1[0] > stain2[0]:
            W = np.array((stain1[:, 0], stain2[:, 0])).T
        else:
            W = np.array((stain2[:, 0], stain1[:, 0])).T

        W = np.clip(W, 0, None)
        # H = np.linalg.lstsq(W, od.T, rcond=None)[0]
        # H = np.clip(H, 0, None)
        # residual = od.T - (W @ H)

        return W


class VahadaneStainExtractor(StainExtractor):
    """
    Vahadane stain extractor
    """

    def __init__(self, Io, lamb, beta1=0, beta2=np.Inf):
        self.Io = Io
        self.lamb = lamb
        self.beta1 = beta1
        self.beta2 = beta2

    def get_stain_matrix(self, img):
        # TODO: Try sklearn.decomposition.DictionaryLearning
        # check if image is blank
        if np.std(img) == 0.0:
            return None, img[0, 0, 0], 0

        # reshape image
        img = img.reshape((-1, 3))

        # calculate optical density
        od = rgb2od(img, self.Io)

        # remove too bright or too dark pixels
        od_hat = od[(np.sum(od ** 2, axis=1) > self.beta1 ** 2) & (np.sum(od ** 2, axis=1) < self.beta2 ** 2)]

        # compute W
        W = spams.trainDL(od_hat.T, K=2, lambda1=self.lamb, mode=2, modeD=0, posAlpha=True, posD=True, verbose=False)
        if W[0, 0] < W[0, 1]:
            W = W[:, [1, 0]]
        # H = np.linalg.lstsq(W, od.T, rcond=None)[0]
        # residual = od.T - (W @ H)

        return W


# def get_stain(file, extractor_type):
#     file_name = os.path.basename(file)
#     img = np.array(Image.open(file))
#     size = img.shape
#     if extractor_type == 'nmf':
#         W, H, _ = get_stain_nmf(img, 255, 0.1)
#     elif extractor_type == 'svd':
#         W, H, _ = get_stain_svd(img, 255)
#
#     # H99 = np.percentile(H, 99, axis=1).reshape((2, 1))
#
#     return (file_name, size, W)