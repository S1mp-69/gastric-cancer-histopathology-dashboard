import numpy as np
from stain_san.stain_extractor import MacenkoStainExtractor, VahadaneStainExtractor
from stain_san.utils import rgb2od

class StainAdaptor:
    """
    Stain adaptor
    Borrows heavily from StainTools (https://github.com/Peter554/StainTools)
    """

    def __init__(self, Io=255, alpha=0.5, lamb=0.1, beta1=0.1, beta2=np.Inf, extract_method='macenko'):
        self.Io = Io
        self.alpha = alpha
        self.lamb = lamb
        self.beta1 = beta1
        self.beta2 = beta2
        if extract_method.lower() == 'macenko':
            self.stain_extractor = MacenkoStainExtractor(self.Io, self.alpha, self.beta1, self.beta2)
        elif extract_method.lower() == 'vahadane':
            self.stain_extractor = VahadaneStainExtractor(self.Io, self.lamb, self.beta1, self.beta2)
        else:
            raise Exception('Extract method not supported.')

    def get_stain_decomp(self, img):
        W = self.get_stain_matrix(img)
        od = rgb2od(img, self.Io)
        H = self.get_intensities(W, od)

        return W, H

    def get_stain_matrix(self, img):
        W = self.stain_extractor.get_stain_matrix(img)

        return W

    @staticmethod
    def get_intensities(W, od):
        H = np.linalg.lstsq(W, od.T, rcond=None)[0]
        H = np.clip(H, 0, None)
        # residual = od.T - (W @ H)

        return H

    @staticmethod
    def get_max_intensities(H):
        H99 = np.percentile(H, 99, axis=1).reshape((2, 1))

        return H99
