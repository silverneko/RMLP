import logging
logger = logging.getLogger(__name__)

import math
import imageio
from numba import jit
import numpy as np
import skimage
from skimage.transform import resize
from scipy import ndimage as ndi

__all__ = ['rmlp']

def _smooth(image, sigma, mode, cval):
    """Return image with each channel smoothed by the Gaussian filter."""
    smoothed = np.empty(image.shape, dtype=np.double)

    # apply Gaussian filter to all channels independently
    ndi.gaussian_filter(image, sigma, output=smoothed,
                        mode=mode, cval=cval)
    return smoothed

def _pyramid_laplacian(image, max_layer=-1, downscale=2, sigma=None, order=1,
                      mode='reflect', cval=0):
    """Yield images of the laplacian pyramid formed by the input image.
    Each layer contains the difference between the downsampled and the
    downsampled, smoothed image::
        layer = resize(prev_layer) - smooth(resize(prev_layer))
    Note that the first image of the pyramid will be the difference between the
    original, unscaled image and its smoothed version. The total number of
    images is `max_layer + 1`. In case all layers are computed, the last image
    is either a one-pixel image or the image where the reduction does not
    change its shape.
    Parameters
    ----------
    image : ndarray
        Input image.
    max_layer : int
        Number of layers for the pyramid. 0th layer is the original image.
        Default is -1 which builds all possible layers.
    downscale : float, optional
        Downscale factor.
    sigma : float, optional
        Sigma for Gaussian filter. Default is `2 * downscale / 6.0` which
        corresponds to a filter mask twice the size of the scale factor that
        covers more than 99% of the Gaussian distribution.
    order : int, optional
        Order of splines used in interpolation of downsampling. See
        `skimage.transform.warp` for detail.
    mode : {'reflect', 'constant', 'edge', 'symmetric', 'wrap'}, optional
        The mode parameter determines how the array borders are handled, where
        cval is the value when mode is equal to 'constant'.
    cval : float, optional
        Value to fill past edges of input if mode is 'constant'.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension. By default, is set to True for
        3D (2D+color) inputs, and False for others. Starting in release 0.16,
        this will always default to False.
    Returns
    -------
    pyramid : generator
        Generator yielding pyramid layers as float images.
    References
    ----------
    .. [1] http://web.mit.edu/persci/people/adelson/pub_pdfs/pyramid83.pdf
    .. [2] http://sepwww.stanford.edu/data/media/public/sep/morgan/texturematch/paper_html/node3.html
    """
    #multichannel = _multichannel_default(multichannel, image.ndim)
    #_check_factor(downscale)
    assert(downscale > 1)

    if sigma is None:
        # automatically determine sigma which covers > 99% of distribution
        sigma = 2 * downscale / 6.0

    layer = 0
    current_shape = image.shape
    out_shape = tuple(
        [math.ceil(d / float(downscale)) for d in current_shape])

    smoothed_image = _smooth(image, sigma, mode, cval)
    yield image - smoothed_image

    # build downsampled images until max_layer is reached or downscale process
    # does not change image size
    while layer != max_layer:
        layer += 1

        resized_image = resize(smoothed_image, out_shape, order=order,
                               mode=mode, cval=cval, anti_aliasing=False)
        smoothed_image = _smooth(resized_image, sigma, mode, cval)

        current_shape = np.asarray(resized_image.shape)
        out_shape = tuple(
            [math.ceil(d / float(downscale)) for d in current_shape])

        last_layer = np.all(current_shape == out_shape) or layer == max_layer-1
        if last_layer:
            yield resized_image
            break
        else:
            yield resized_image - smoothed_image

def pyramid_fusion(images, M, K):
    """
    Fused pyramid layers using the mask.

    Parameters
    ----------
    I : np.ndarray
        The source image.
    M : np.ndarray
        The masked image.
    K : int
        Level of the pyramid.
    """
    # automatically determine sigma which covers > 99% of distribution
    downscale = 2
    sigma = 2 * downscale / 6.0

    LP = zip(*[list(_pyramid_laplacian(img, max_layer=K)) for img in images])
    F = []

    for lp in LP:
        fused = np.zeros_like(lp[0])
        M = resize(M, lp[0].shape, order=0, mode='edge', anti_aliasing=False)
        for (i, l) in zip(range(1, 1+len(lp)), lp):
            fused[M == i] = l[M == i]
        F.append(fused)

    fused_F = F[-1]
    for f in reversed(F[:-1]):
        assert(all(i <= j for (i, j) in zip(fused_F.shape, f.shape)))
        resized_F = resize(fused_F, f.shape, order=1,
                           mode='edge', anti_aliasing=False)
        smoothed_F = _smooth(resized_F, sigma=sigma, mode='reflect', cval=0)
        fused_F = smoothed_F + f

    return fused_F

@jit
def _generate_seeds(D, t=0.5):
    """
    Find seed pixels by density distributions.

    Parameters
    ----------
    D : np.ndarray
        Density distribution of supplied images.
    t : float
        Threshold for seed pixels, default to 0.5
    """
    S = D[0].copy()
    for d in D[1:]:
        S[d > S] = d[d > S]
    return S > t

@jit
def _density_distribution(n, M, r):
    """
    Calculate density distribution along specified circular regions.

    Parameters
    ----------
    n : int
        Number of blurred images.
    M : np.ndarray
        The mask image.
    r : float
        Radius of circle that counted as spatial neighborhood.
    """
    D = []
    mp = np.zeros((2*r+1, 2*r+1)) # buffer area
    r2 = r*r
    c = 1. / (np.pi * r2) # normalize factor
    for _n in range(1, n+1):
        Dp = np.empty_like(M)
        # delta function
        Mp = (M == _n)
        n, m = M.shape
        for y in range(0, n):
            for x in range(0, m):
                v = 0
                pu = min(y+r, n-1)
                pd = max(y-r, 0)
                pr = min(x+r, m-1)
                pl = max(x-r, 0)
                for yy in range(pd, pu+1):
                    for xx in range(pl, pr+1):
                        if Mp[yy, xx] and ((xx-x)*(xx-x) + (yy-y)*(yy-y) <= r2):
                            v += 1
                Dp[y, x] = v * c
        D.append(Dp)
    return D

def dbrg(n, M, r):
    """
    Segmentation by density-based region growing (DBRG).

    Parameters
    ----------
    n : int
        Number of blurred images.
    M : np.ndarray
        The mask image.
    """
    D = _density_distribution(n, M, r)
    for i, d in enumerate(D):
        imageio.imwrite("data/D{}.tif".format(i), d.astype(np.float32))
    S = _generate_seeds(D)

    # unlabeled
    R = np.zeros_like(M)
    V = np.full(M.shape, np.NINF, dtype=np.float32)

    # label by density map
    for i, d in enumerate(D):
        R[(d > V) & S] = i+1
        V[(d > V) & S] = d[(d > V) & S]

    # label by density connectivity #TODO
    n, m = M.shape
    v = []
    for y in range(0, n):
        for x in range(0, m):
            v = 0
            pu = min(y+r, n-1)
            pd = max(y-r, 0)
            pr = min(x+r, m-1)
            pl = max(x-r, 0)
            for yy in range(pd, pu+1):
                for xx in range(pl, pr+1):
                    if Mp[yy, xx] and ((xx-x)*(xx-x) + (yy-y)*(yy-y) <= r2):
                        v += 1
            Dp[y, x] = v * c
    imageio.imwrite("data/R.tif", R)

    raise RuntimeError

    V = np.full_like(M, np.NINF)
    n, m = M.shape
    for i, d in enumerate(D):
        R[d > V] = i+1
        V[d > V] = d[d > V]

    #TODO process unlabeled pixels

    return R

@jit
def _modified_laplacian(I):
    """Calculate modified Laplacian."""
    J = np.empty_like(I)
    n, m = I.shape
    for y in range(0, n):
        for x in range(0, m):
            pc = I[y, x]
            # symmetric padding, pad size 1
            pu = I[y+1, x] if y < n-1 else I[y-1, x]
            pd = I[y-1, x] if y > 0 else I[y+1, x]
            pr = I[y, x+1] if x < m-1 else I[y, x-1]
            pl = I[y, x-1] if x > 0 else I[y, x+1]
            J[y, x] = abs(2*pc - pl - pr) + abs(2*pc - pu - pd)
    return J

@jit
def sml(I, T):
    """
    Calculate summed-modified-Laplacian (SML) measurement.

    Parameters
    ----------
    I : np.ndarray
        Input image.
    T : float
        The discrimination threshold, optimal value of T varies in [0, 10].
    """
    G = _modified_laplacian(I)
    Gp = np.zeros((3, 3)) # buffer area
    S = np.empty_like(I)
    n, m = S.shape
    for y in range(0, n):
        for x in range(0, m):
            pu = min(y+1, n-1)
            pd = max(y-1, 0)
            pr = min(x+1, m-1)
            pl = max(x-1, 0)
            Gp = G[pd:pu+1, pl:pr+1]
            S[y, x] = np.sum(Gp[Gp >= T])
    return S

def _generate_init_mask(images, T):
    S = []
    for image in images:
        S.append(sml(image, T))

    M = np.full(images[0].shape, -1, dtype=np.uint32)
    V = np.full_like(images[0], np.NINF)
    n, m = S[0].shape
    for i, s in enumerate(S):
        M[abs(s) > V] = i+1
        V[abs(s) > V] = s[abs(s) > V]
    return M

def rmlp(images, T=0.007):
    """
    Perform region-based Laplacian pyramids multi-focus image fusion.
    """
    M = _generate_init_mask(images, T)
    R = dbrg(len(images), M, 2)
    imageio.imwrite("data/M.tif", M)
    F = pyramid_fusion(images, R, 3)
    return F
