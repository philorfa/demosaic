'''Helper functions for CFA'''
import numpy as np
import itertools as itt
import functools
import os


fp_img = np.float32


# basic input output
def pics_in_dir(dirpath, ending=''):
    pic_ext = ('.png', '.jpg', '.tif', '.PNG', '.JPG', '.TIF')
    flist = os.listdir(dirpath)
    flist = filter(lambda x: x.endswith(pic_ext) and (ending in x), flist)
    return [os.path.join(dirpath, fl) for fl in flist]


# convert all image inputs to FP, with range 0. to 1.
def img2fp(func):
    @functools.wraps(func)
    def img2fp_func(*args, **kwargs):
        # assume first argument is the image
        img = convert2fp(args[0])
        args = (img, *args[1:])
        return func(*args, **kwargs)
    return img2fp_func


def convert2fp(img):
    dtype = img.dtype
    if dtype != fp_img:
        if np.issubdtype(dtype, np.integer):
            px_max = np.float64(np.iinfo(dtype).max)
            img = img.astype(np.float64) / px_max
            img = img.astype(fp_img)
        elif np.issubdtype(dtype, np.floating):
            img = img.astype(fp_img)
        np.clip(img, fp_img(0.), fp_img(1.), out=img)
    return img


def convert2int(img, dtype):
    if np.issubdtype(img.dtype, np.integer):
        return img
    if not np.issubdtype(dtype, np.integer):
        raise ValueError('Requested dtype should be integer')
    img *= np.iinfo(dtype).max
    return img.astype(dtype)


# mosaic and helper functions
def _plex_channels(imgrgb, rch, gch, bch, cfa, order):
    if cfa == 'bayer':
        stride = 2
    elif cfa == 'tetra':
        stride = 4
    elif cfa == 'nona':
        stride = 6
    elif cfa == 'chame':
        stride = 8
    else:
        raise ValueError('Unknown CFA ' + str(cfa))

    strd_half = stride // 2
    ordr = order.lower()
    range1 = range(strd_half)
    range2 = range(strd_half, stride)
    if ordr == 'rggb':
        ritr = [range1, range1]
        bitr = [range2, range2]
        gitr1 = [range1, range2]
        gitr2 = [range2, range1]
    elif ordr == 'grbg':
        ritr = [range1, range2]
        bitr = [range2, range1]
        gitr1 = [range1, range1]
        gitr2 = [range2, range2]
    elif ordr == 'gbrg':
        ritr = [range2, range1]
        bitr = [range1, range2]
        gitr1 = [range1, range1]
        gitr2 = [range2, range2]
    elif ordr == 'bggr':
        ritr = [range2, range2]
        bitr = [range1, range1]
        gitr1 = [range1, range2]
        gitr2 = [range2, range1]
    else:
        raise ValueError('Unknown pixel ordering. Code it yourself')

    for ikj in itt.product(*ritr):
        ik, ij = ikj
        imgrgb[ik::stride, ij::stride, 0] = rch[ik::stride, ij::stride]
    for ikj in itt.product(*bitr):
        ik, ij = ikj
        imgrgb[ik::stride, ij::stride, 2] = bch[ik::stride, ij::stride]
    for ikj in itt.chain(itt.product(*gitr1), itt.product(*gitr2)):
        ik, ij = ikj
        imgrgb[ik::stride, ij::stride, 1] = gch[ik::stride, ij::stride]


def mosaic(img, cfa='bayer', order='RGGB', one_channel=False):
    img_out = np.zeros_like(img)
    _plex_channels(img_out, img[:, :, 0], img[:, :, 1],
                   img[:, :, 2], cfa, order)
    if one_channel:
        img_out = img_out.sum(axis=2, dtype=img.dtype)
    return img_out


def rgbmosaic2_ch1(img):
    return img.sum(axis=2, dtype=img.dtype)


def ch1mosaic2_rgb(img, cfa='bayer', order='RGGB'):
    shp = (img.shape[0], img.shape[1], 3)
    img_out = np.zeros(shp, dtype=img.dtype)
    _plex_channels(img_out, img, img, img, cfa, order)
    return img_out


def cfa_mask(img, cfa='bayer', order='RGGB'):
    img_out = np.ones_like(img, dtype=bool)
    return mosaic(img_out, cfa, order, one_channel=False)


# metric functions
def cpsnr(original, estimate):
    dtype = original.dtype
    if np.issubdtype(dtype, np.integer):
        pixel_max = np.iinfo(dtype).max
        origf = original.astype(float)
        estf = estimate.astype(float)
    else:
        pixel_max = 1.
        origf = original
        estf = estimate
    mse = np.mean(np.square(origf - estf))
    if mse == 0.:
        return 100.
    return 20. * np.log10(pixel_max / np.sqrt(mse))
