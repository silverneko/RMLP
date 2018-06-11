from os.path import join

from imageio import imwrite
import numpy as np
from skimage.draw import circle
from skimage.filters import gaussian
from skimage.util import img_as_float

# init seed
np.random.seed(0xBEEFCA9E)

def _checkerboard(shape, k, n=2, r=100, sigma=2):
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
        Radius of the blurred region.
    sigma : float
        Sigma of the blur kernel.
    """
    if any([x % k for x in shape]):
        raise ValueError("shape is not divisible by k")

    Igt = np.empty(shape=shape, dtype=np.uint8)
    bs = (shape[0]//k, shape[1]//k)
    for j in range(k):
        for i in range(k):
            # baseline intensity
            v = 64 if (i+j) % 2 else 192
            # perturbation
            v += np.random.randint(-30, 31)
            Igt[bs[0]*j:bs[0]*(j+1), bs[1]*i:bs[1]*(i+1)] = v

    # blurred
    Ib = gaussian(Igt, sigma=sigma, preserve_range=True)
    Ibs = []
    c = []
    i = 0
    while i < n:
        # buffer
        Ibb = Igt.copy()

        px = np.random.randint(r, shape[0]-r)
        py = np.random.randint(r, shape[1]-r)

        if any([((p[0]-px)*(p[0]-px) + (p[1]-py)*(p[1]-py)) < r*r for p in c]):
            # regenerate
            pass
        else:
            yy, xx = circle(py, px, r)
            Ibb[yy, xx] = Ib[yy, xx]

            # record result
            Ibs.append(Ibb)
            c.append((py, px))
            # next image
            i += 1

    return Igt, Ibs

if __name__ == '__main__':
    gt, bl = _checkerboard((512, 512), 32, n=3)

    root = "data"
    imwrite(join(root, "ground_truth.png"), gt)
    for i, im in enumerate(bl):
        imwrite(join(root, "{:03d}.png".format(i)), im)
