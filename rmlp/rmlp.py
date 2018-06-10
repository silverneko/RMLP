import logging
logger = logging.getLogger(__name__)

import imageio
import numpy as np
import scipy

__all__ = ['rmlp']

def _gaussian_pyramid(I, K):
    pass

def pyramid_fusion(I, K):
    """
    Fused by pyramid layers.

    Parameters
    ----------
    I : np.ndarray
        The source image.
    K : int
        Level of the pyramid.
    """
    G = _gaussian_pyramid(I, K)
    pass

def _density_distribution(M, r):
    """
    Calculate density distribution along specified circular regions.

    Parameters
    ----------
    M : np.ndarray
        The mask image.
    r : float
        Radius of circle that counted as spatial neighborhood.
    """
    pass

def dbrg(M, r):
    """
    Segmentation by density-based region growing (DBRG).

    Parameters
    ----------
    M : np.ndarray
        The mask image.
    """
    D = _density_distribution(M, R)
    R = None
    return R

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

def sml(I, T, w=1):
    """
    Calculate summed-modified-Laplacian (SML) measurement.

    Parameters
    ----------
    I : np.ndarray
        Input image.
    T : float
        The discrimination threshold, optimal value of T varies in [0, 10].
    w : int, default to 1
        Window size when computing the sum.
    """
    G = _modified_laplacian(I)
    S = np.empty_like(I)
    n, m = S.shape
    for y in range(0, n):
        for x in range(0, m):
            pass

def rmlp(images):
    """
    Perform region-based Laplacian pyramids multi-focus image fusion.
    """
    res = _modified_laplacian(images[0])
    print(res[4:7, 4:7])
    imageio.imwrite("data/test.tif", res)
