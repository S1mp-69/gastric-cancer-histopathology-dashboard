import os
import numpy as np

from joblib import load
from tqdm import tqdm

stain = load('stain.joblib')

for file_name, W, H99 in tqdm(stain):
    img_file = os.path.join(input_dir, file_name)
    img = np.array(Image.open(img_file))
    img.reshape((-1, 3))
    od = rgb2od(img)
    H = np.linalg.lstsq(W, od.T, rcond=None)[0]
    projected_image = stain2image(W, H, size=size)
    Image.fromarray(adapted_img).save(os.path.join(output_dir,
                                                   file_name))
