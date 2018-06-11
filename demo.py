import logging
import os

import imageio
import numpy as np
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
    gt = imageio.imread(gt)
    gt = img_as_float32(gt)
    bl = [imageio.imread(f) for f in bl]
    bl = [img_as_float32(f) for f in bl]

    # sanity check
    s_gt = gt.shape
    if any([im.shape != s_gt for im in bl]):
        raise ValueError("blurred image has a different size")
    return gt, bl

def demo(root, T=None):
    gt, bl = load_dataset_images(root)
    res = rmlp(bl, T)
    return res

if __name__ == '__main__':
    #demo("data/square")
<<<<<<< HEAD
    res = demo("data/checkerboard")
    imsave("data/result.png", res)
=======
    res = demo("data/checkerboard", T=7/255.)
    imsave('res.png', res)
>>>>>>> b04df8b6d05071673cb0198ec057438d22a88df4
