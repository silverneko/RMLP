import logging
import os
import traceback

import imageio
import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave
from skimage.measure import compare_ssim as ssim
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
        logger.debug("reading '{}'".format(f))
        I = imageio.imread(f)
        if I.ndim == 3:
            I = rgb2gray(I)
        I = img_as_float32(I)
        bl[i] = I

    if gt is not None:
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
    result = rmlp(bl)

    # calculate measures if ground truth exists
    if gt is not None:
        for i, bb in enumerate(bl):
            v_mse = np.linalg.norm(bb - gt)
            v_ssim = ssim(bb, gt, data_range=(bb.max()-bb.min()))
            logger.info("blurred {}".format(i))
            logger.info(".. MSE = {:.4f}, SSIM = {:.4f}".format(v_mse, v_ssim))
        v_mse = np.linalg.norm(result - gt)
        v_ssim = ssim(result, gt, data_range=(result.max()-result.min()))
        logger.info("result")
        logger.info(".. MSE = {:.4f}, SSIM = {:.4f}".format(v_mse, v_ssim))

    return result

if __name__ == '__main__':
    root = "data/mt"
    try:
        result = demo(root)

        # convert to uint8 for preview
        result = rescale_intensity(result, out_range=(0, 2**8-1))
        result = result.astype(np.uint8)
        imageio.imwrite(os.path.join(root, "result.png"), result)
    except Exception as e:
        logger.error(traceback.format_exc())
