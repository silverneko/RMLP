import os

import imageio
import numpy as np
from skimage.io import imread, imsave
from skimage.util import img_as_float

from rmlp import rmlp

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
    gt = imageio.imread(gt)
    gt = img_as_float(gt)
    bl = [imageio.imread(f) for f in bl]
    bl = [img_as_float(f) for f in bl]

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
    res = demo("data/checkerboard")
    imsave('res.png', res)
