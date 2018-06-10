from os.path import join

from imageio import imwrite
import numpy as np
import skimage
import skimage.filters
from skimage.util import img_as_float

# init seed
np.random.seed(0xBEEFCA9E)

def _checkerboard(shape, k, n=2, r=100):
    """
    Generate checkerboard pattern test dataset.

    Parameters
    ----------
    shape : tuple of int
        Desired image size, column major.
    k : int
        Number of blocks in the pattern along each dimension.
    n : int
        Number of blurred regions in the image.
    r : int
        Blurred radius.
    """
    if any([x % k for x in shape]):
        raise ValueError("shape is not divisible by k")

    Igt = np.empty(shape=shape, dtype=np.uint8)
    bs = (shape[0]//2, shape[1]//2)
    for j in range(k):
        for i in range(k):
            # baseline intensity
            v = 64 if (i+j) % 2 else 192
            # perturbation
            v += np.random.randint(-30, 31)
            I[bs[0]*j:bs[0]*(j+1), bs[1]*i:bs[1]*(i+1)] = v

    # ground truth
    Igt = img_as_float(Igt)


    p1 = np.copy(Igt)
    p2 = np.copy(Igt)

    rr, cc = skimage.draw.circle(128, 128, 100)
    p1[rr, cc] = bimg[rr, cc]
    rr, cc = skimage.draw.circle(384, 384, 100)
    p2[rr, cc] = bimg[rr, cc]

    bimg = skimage.filters.gaussian(img, sigma=4)

if __name__ == '__main__':
    gt, bl = _checkerboard((512, 512), 32)

    root = "data"
    imwrite(join(root, "ground_truth.png"), gt)
    for i, im in enumerate(bl):
        imwrite(join(root, "{:03d}.png".format(i)), im)
