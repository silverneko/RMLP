import logging
import os

import imageio
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from skimage.util import img_as_float32

from rmlp import rmlp

#####

handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(levelname).1s %(asctime)s %(message)s', '%H:%M:%S'
)
handler.setFormatter(formatter)
logging.basicConfig(level=logging.DEBUG, handlers=[handler])
logger = logging.getLogger(__name__)

#####

def list_dataset_images(root):
    ground_truth = None
    blurred = []
    for file_name in os.listdir(root):
        file_path = os.path.join(root, file_name)
        if os.path.isfile(file_path):
            if file_name.startswith('ground_truth'):
                if ground_truth:
                    raise ValueError("duplicated ground truth found")
                ground_truth = file_path
            else:
                blurred.append(file_path)
    if len(blurred) < 2:
        raise ValueError("insufficient blurred image in the dataset")
    return ground_truth, blurred

def load_dataset_images(root):
    gt, bl = list_dataset_images(root)

    for i, f in enumerate(bl):
        I = imageio.imread(f)
        if I.ndim == 3:
            I = rgb2gray(I)
        I = img_as_float32(I)
        bl[i] = I

    if gt:
        gt = imageio.imread(gt)
        if gt.ndim == 3:
            gt = rgb2gray(gt)
        gt = img_as_float32(gt)

        # sanity check
        s_gt = gt.shape
        if any([im.shape != s_gt for im in bl]):
            raise ValueError("blurred image has a different size")

    return gt, bl

def demo(root):
    gt, bl = load_dataset_images(root)
    res = rmlp(bl)
    return res

if __name__ == '__main__':
    #demo("data/square")
    root = "data/slika2"
    res = demo(root)
    imsave(os.path.join(root, "result.bmp"), res)
