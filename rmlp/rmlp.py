import logging
logger = logging.getLogger(__name__)

import imageio
from numba import jit
import numpy as np
from skimage.transform import resize

__all__ = ['rmlp']

def _laplacian_pyramid(G):
    K = len(G)
    LP = []
    for k in range(0, K-1):
        # upscale lower level
        kp = resize(G[k+1], (k.shape[0]*2, k.shape[1]*2))
        # Laplacian by difference of Gaussians
        LP.append(G[k] - kp)
    LP.append(G[K-1])
    return LP

def _gaussian_pyramid(I, K):
    pass

def pyramid_fusion(images, M, K):
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
    LP = _laplacian_pyramid(G)

    #TODO fuse LP
    pass

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
    # mark seeds
    for i, d in enumerate(D):
        D[i] = D[i] > .5

    # unlabeled
    R = np.full_like(M, -1)
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

    M = np.full_like(images[0], -1)
    V = np.full_like(images[0], np.NINF)
    n, m = S[0].shape
    for i, s in enumerate(S):
        M[abs(s) > V] = i+1
        V[abs(s) > V] = s[abs(s) > V]
    return M

def rmlp(images, T=7):
    """
    Perform region-based Laplacian pyramids multi-focus image fusion.
    """
    M = _generate_init_mask(images, T)
    R = dbrg(len(images), M, 2)
    imageio.imwrite("data/R.tif", R.astype(np.uint8))
    F = pyramid_fusion(images, R, 3)
