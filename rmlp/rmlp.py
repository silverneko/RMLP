import logging
logger = logging.getLogger(__name__)

import imageio
from numba import jit
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
    for _n in range(n):
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
    # mark seeds
    for i, d in enumerate(D):
        D[i] = D[i] > .5
    for i, d in enumerate(D):
        imageio.imwrite("data/D{}.tif".format(i), d.astype(np.uint8))
    R = None
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

    M = np.full_like(images[0], -1)
    V = np.full_like(images[0], np.NINF)
    n, m = S[0].shape
    for i, s in enumerate(S):
        M[abs(s) > V] = i
        V[abs(s) > V] = s[abs(s) > V]
    return M

def rmlp(images, T=7):
    """
    Perform region-based Laplacian pyramids multi-focus image fusion.
    """
    M = _generate_init_mask(images, T)
    R = dbrg(len(images), M, 2)
