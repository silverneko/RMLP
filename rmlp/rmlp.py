import logging
logger = logging.getLogger(__name__)

import math
import imageio
from numba import jit
import numpy as np
import skimage
import skimage.draw
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

def pyramid_fusion(images, M, K, sigma=None):
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
    if sigma is None:
        downscale = 2
        sigma = 2 * downscale / 6.0

    LP = zip(*[list(_pyramid_laplacian(img, max_layer=K, sigma=sigma)) for img in images])
    F = []

    for lp in LP:
        fused = np.zeros_like(lp[0])
        M = resize(M, lp[0].shape, preserve_range=True,
                   order=0, mode='edge', anti_aliasing=False)
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

    rr, cc = skimage.draw.circle(r, r, r+1)
    for _n in range(1, n+1):
        Dp = np.zeros(M.shape)
        # delta function
        Mp = (M == _n)
        Mp = np.pad(Mp, [(r, r), (r, r)], mode='constant')
        Ar = np.pad(np.ones(Mp.shape), [(r, r), (r, r)], mode='constant')
        n, m = M.shape
        for y in range(0, n):
            for x in range(0, m):
                yy = rr + y
                xx = cc + x
                v = np.sum(Mp[yy, xx])
                c = np.sum(Ar[yy, xx])
                Dp[y, x] = 1.0 * v / c
        D.append(Dp)
    return D

def dbrg(images, T, r):
    """
    Segmentation by density-based region growing (DBRG).

    Parameters
    ----------
    n : int
        Number of blurred images.
    M : np.ndarray
        The mask image.
    r : int
        Density connectivity search radius.
    """
    n = len(images)
    M = _generate_init_mask(images, T)
    D = _density_distribution(n, M, r)
    S = _generate_seeds(D)

    # unlabeled
    R = np.full(M.shape, 0, dtype=np.uint32)
    V = np.full(M.shape, np.NINF, dtype=np.float32)

    # label by density map
    for i, d in enumerate(D):
        R[(d > V) & S] = i+1
        V[(d > V) & S] = d[(d > V) & S]

    # label by density connectivity
    n, m = M.shape
    v = np.empty(len(D)+1, dtype=np.float32)
    ps = [] # reset of the pixel coordinates
    for y in range(0, n):
        for x in range(0, m):
            if R[y, x] > 0:
                continue
            pu = min(y+r, n-1)
            pd = max(y-r, 0)
            pr = min(x+r, m-1)
            pl = max(x-r, 0)
            v.fill(0)
            for yy in range(pd, pu+1):
                for xx in range(pl, pr+1):
                    if ((xx-x)*(xx-x) + (yy-y)*(yy-y) <= r*r):
                        v[R[yy, xx]] += 1
            R[y, x] = v.argmax()
            if R[y, x] == 0:
                ps.append((y, x))

    # label by nearest neighbor
    psv = [] # filled result
    for y, x in ps:
        r = 1
        while True:
            pu = min(y+r, n-1)
            pd = max(y-r, 0)
            pr = min(x+r, m-1)
            pl = max(x-r, 0)
            v = []
            for yy in range(pd, pu+1):
                for xx in range(pl, pr+1):
                    if R[yy, xx] > 0:
                        v.append((R[yy, xx], (xx-x)*(xx-x) + (yy-y)*(yy-y)))
            if len(v) == 0:
                r += 1
            else:
                v.sort(key=lambda p: p[1])
                psv.append(v[0][0])
                break
    for (y, x), v in zip(ps, psv):
        R[y, x] = v

    assert(np.all(R != 0))
    return R

@jit
def _modified_laplacian(I):
    """
    Calculate modified Laplacian.

    Parameters
    ----------
    I : np.ndarray
        Supplied raw image.
    """
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
    """
    Generate mask estimation based on SML.

    Parameters
    ----------
    images : list of np.ndarray
        List of original raw images.
    T : float
        Blur level criteria.
    """
    S = []
    for image in images:
        S.append(sml(image, T))

    M = np.full(images[0].shape, 0, dtype=np.uint32)
    V = np.full_like(images[0], np.NINF)
    n, m = S[0].shape
    for i, s in enumerate(S):
        M[abs(s) > V] = i+1
        V[abs(s) > V] = s[abs(s) > V]
    return M

def rmlp(images, T=1/255., r=4, K=7):
    """
    Perform region-based Laplacian pyramids multi-focus image fusion.

    Parameters
    ----------
    images : list of np.ndarray
        Blurred images.
    T : float
        Initial mask threshold.
    r : int
        Density connectivity search radius.
    K : int
        Level of the pyramids.
    """
    R = dbrg(images, T, r)
    F = pyramid_fusion(images, R, K)
    return F
