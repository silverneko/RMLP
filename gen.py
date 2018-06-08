#!/usr/bin/env python3

import numpy as np
import skimage
import skimage.filters
from skimage.io import imsave
from skimage.util import img_as_float
from math import floor

np.random.seed(0xBEEFCA9E)

img = np.zeros((512, 512), dtype=np.uint8)

for i in range(256):
    img[:,2*i:2*i+2] = i
for i in range(256):
    img[128:384, 128+i] = 255 - i

img = img_as_float(img)
bimg = skimage.filters.gaussian(img, sigma=4)

idx = np.linspace(0, 512, 6, dtype=int)

p1 = np.copy(img)
p2 = np.copy(img)

p1[idx[1]:idx[2]] = bimg[idx[1]:idx[2]]
p2[idx[3]:idx[4]] = bimg[idx[3]:idx[4]]

imsave('gta.png', img)
imsave('pa1.png', p1)
imsave('pa2.png', p2)

img = np.zeros((512, 512), dtype=np.uint8)
for i in range(16):
    for j in range(16):
        if (i+j) % 2 == 1:
            color = 64
        else:
            color = 192
        perturb = np.random.randint(-30, 31)
        img[32*i:32*i+32, 32*j:32*j+32] = color + perturb
img = img_as_float(img)
bimg = skimage.filters.gaussian(img, sigma=4)

p1 = np.copy(img)
p2 = np.copy(img)

rr, cc = skimage.draw.circle(128, 128, 100)
p1[rr, cc] = bimg[rr, cc]
rr, cc = skimage.draw.circle(384, 384, 100)
p2[rr, cc] = bimg[rr, cc]

imsave('gtb.png', img)
imsave('pb1.png', p1)
imsave('pb2.png', p2)
