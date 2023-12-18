'''Debayering functions'''
import numpy as np
import scipy.ndimage as spimg
import cfa_image as cfa


# simple intepolation
def bilinear_interpolation(image):
    image_out = np.zeros_like(image)
    image_out.astype(float)
    krnl_rb = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])
    krnl_gg = np.array([[0., 1., 0.], [1., 4., 1.], [0., 1., 0.]])
    krnl_rb *= .25
    krnl_gg *= .25
    image_out[:, :, 0] = spimg.convolve(image[:, :, 0], krnl_rb, mode='mirror')
    image_out[:, :, 1] = spimg.convolve(image[:, :, 1], krnl_gg, mode='mirror')
    image_out[:, :, 2] = spimg.convolve(image[:, :, 2], krnl_rb, mode='mirror')
    return image_out.astype(image.dtype)


# adaptive interpolation
def _dir_grad(img, threshold, gfilter):
    # the gradients
    if img.ndim > 2:
        imgf = cfa.rgbmosaic2_ch1(img)
    else:
        imgf = img
    krnlg = np.array([1., 0., -1.])
    gradh = spimg.convolve1d(imgf, krnlg, mode='mirror')
    gradv = spimg.convolve1d(imgf, krnlg, axis=0, mode='mirror')
    gradh = np.absolute(gradh)
    gradv = np.absolute(gradv)
    if gfilter:
        delh, delv = _grad_filter(gradh, gradv)
    else:
        delh = gradh
        delv = gradv
    maskh = delv > delh + threshold
    maskv = delh > delv + threshold
    maskb = np.logical_and(np.logical_not(maskh), np.logical_not(maskv))
    return maskh, maskv, maskb


def _grad_filter(gradh, gradv):
    krnl_del = np.array([[0., 0., 1., 0., 1.],
                         [0., 0., 0., 1., 0.],
                         [0., 0., 3., 0., 3.],
                         [0., 0., 0., 1., 0.],
                         [0., 0., 1., 0., 1.]])
    delh = spimg.convolve(gradh, krnl_del, mode='constant')
    delv = spimg.convolve(gradv, krnl_del.T, mode='constant')
    return delh, delv


def _chrominance_grad(grnh, grnv, imgf, mask, threshold, gfilter):
    chromh = np.zeros_like(imgf)
    chromv = np.zeros_like(imgf)
    mask = np.logical_or(mask[:, :, 0], mask[:, :, 2])
    chromh[mask] = imgf[mask] - grnh[mask]
    chromv[mask] = imgf[mask] - grnv[mask]

    gradh = np.pad(chromh, ((0, 0), (0, 2)), mode='reflect')[:, 2:]
    gradv = np.pad(chromv, ((0, 2), (0, 0)), mode='reflect')[2:, :]
    gradh = np.absolute(chromh - gradh)
    gradv = np.absolute(chromv - gradv)

    if gfilter:
        delh, delv = _grad_filter(gradh, gradv)
    else:
        delh = gradh
        delv = gradv

    maskh = delv > delh + threshold
    maskv = delh > delv + threshold
    maskb = np.logical_and(np.logical_not(maskh), np.logical_not(maskv))
    return maskh, maskv, maskb


def adaptive_interpolation(img, classifier='grad', img_class=None,
                           threshold=0., order='RGGB', gfilter=True):
    msk_px = cfa.cfa_mask(img, order=order)
    imgf = cfa.rgbmosaic2_ch1(img)

    # green channel interpolation
    krnli = np.array([.5, 0., .5])
    krnlirb = np.array([-.25, 0., .5, 0., -.25])
    grnh = spimg.convolve1d(imgf, krnli, mode='mirror')
    grnh += spimg.convolve1d(imgf, krnlirb, mode='mirror')
    grnh[msk_px[:, :, 1]] = img[msk_px[:, :, 1], 1]
    grnv = spimg.convolve1d(imgf, krnli, axis=0, mode='mirror')
    grnv += spimg.convolve1d(imgf, krnlirb, axis=0, mode='mirror')
    grnv[msk_px[:, :, 1]] = img[msk_px[:, :, 1], 1]

    if img_class is not None:
        imgcl = img_class
    else:
        imgcl = img

    if classifier == 'grad':
        maskh, maskv, maskb = _dir_grad(imgcl, threshold, gfilter)
    elif classifier == 'chrom_grad':
        maskh, maskv, maskb = _chrominance_grad(grnh, grnv, imgf, msk_px,
                                                threshold=threshold,
                                                gfilter=gfilter)

    img_out = np.zeros_like(img)
    img_out[maskh, 1] = grnh[maskh]
    img_out[maskv, 1] = grnv[maskv]
    img_out[maskb, 1] = .5 * grnh[maskb] + .5 * grnv[maskb]

    # red and blue interpolation at green pixels
    grnh = spimg.convolve1d(img_out[:, :, 1], krnli, mode='mirror')
    grnv = spimg.convolve1d(img_out[:, :, 1], krnli, axis=0, mode='mirror')

    rshp = (1, img.shape[1])
    rrow = np.tile(np.expand_dims(msk_px[:, :, 0].any(axis=1), axis=1), rshp)
    brow = np.tile(np.expand_dims(msk_px[:, :, 2].any(axis=1), axis=1), rshp)

    mask = np.logical_and(rrow, msk_px[:, :, 1])
    img_out[mask, 0] = img[mask, 1] - grnh[mask] \
        + spimg.convolve1d(img[:, :, 0], krnli, mode='mirror')[mask]
    img_out[mask, 2] = img[mask, 1] - grnv[mask] \
        + spimg.convolve1d(img[:, :, 2], krnli, axis=0, mode='mirror')[mask]

    mask = np.logical_and(brow, msk_px[:, :, 1])
    img_out[mask, 0] = img[mask, 1] - grnv[mask] \
        + spimg.convolve1d(img[:, :, 0], krnli, axis=0, mode='mirror')[mask]
    img_out[mask, 2] = img[mask, 1] - grnh[mask] \
        + spimg.convolve1d(img[:, :, 2], krnli, mode='mirror')[mask]

    # horizontal and vertical red and blue channels
    redh = spimg.convolve1d(img_out[:, :, 0], krnli, mode='mirror')
    redv = spimg.convolve1d(img_out[:, :, 0], krnli, axis=0, mode='mirror')
    blueh = spimg.convolve1d(img_out[:, :, 2], krnli, mode='mirror')
    bluev = spimg.convolve1d(img_out[:, :, 2], krnli, axis=0, mode='mirror')

    # red interpolation at blue pixels
    mask = np.logical_and(brow, msk_px[:, :, 2])
    mask_cl = np.logical_and(mask, maskh)
    img_out[mask_cl, 0] = img[mask_cl, 2] + redh[mask_cl] - blueh[mask_cl]
    mask_cl = np.logical_and(mask, maskv)
    img_out[mask_cl, 0] = img[mask_cl, 2] + redv[mask_cl] - bluev[mask_cl]
    mask_cl = np.logical_and(mask, maskb)
    img_out[mask_cl, 0] = img[mask_cl, 2] + .5 * redh[mask_cl] \
        - .5 * blueh[mask_cl] + .5 * redv[mask_cl] - .5 * bluev[mask_cl]

    # blue interpolation at red pixels
    mask = np.logical_and(rrow, msk_px[:, :, 0])
    mask_cl = np.logical_and(mask, maskh)
    img_out[mask_cl, 2] = img[mask_cl, 0] + blueh[mask_cl] - redh[mask_cl]
    mask_cl = np.logical_and(mask, maskv)
    img_out[mask_cl, 2] = img[mask_cl, 0] + bluev[mask_cl] - redv[mask_cl]
    mask_cl = np.logical_and(mask, maskb)
    img_out[mask_cl, 2] = img[mask_cl, 0] + .5 * blueh[mask_cl] \
        - .5 * redh[mask_cl] + .5 * bluev[mask_cl] - .5 * redv[mask_cl]

    # fill the known pixels
    img_out[msk_px[:, :, 0], 0] = img[msk_px[:, :, 0], 0]
    img_out[msk_px[:, :, 2], 2] = img[msk_px[:, :, 2], 2]
    return img_out


# linear demosaic
def fft_filter(img, fxy, sigmaxy, shape='ellipse'):
    img_fft = np.fft.fft2(img.astype(float))
    freqx = np.fft.fftfreq(img.shape[1])
    freqy = np.fft.fftfreq(img.shape[0])
    freqx, freqy = np.meshgrid(freqx, freqy)

    hfilt = np.ones_like(img)
    for ff, sgm in zip(fxy, sigmaxy):
        if not isinstance(ff, list):
            ff = [ff, ] * 2
        if not isinstance(sgm, list):
            sgm = [sgm, ] * 2
        sgm_den = sgm[0] * sgm[0]
        dfx = np.square(np.absolute(freqx) - ff[0]) / sgm_den
        sgm_den = sgm[1] * sgm[1]
        dfy = np.square(np.absolute(freqy) - ff[1]) / sgm_den
        if shape == 'line':
            hfilt *= (1. - np.exp(-dfx)) * (1. - np.exp(-dfy))
        elif shape == 'ellipse':
            dfx += dfy
            hfilt *= 1. - np.exp(-dfx)
        else:
            raise ValueError('Unknown FT filter shape', str(shape))
    return np.fft.ifft2(hfilt * img_fft).real.astype(np.float32)


def _luminance_filter(img):
    dtype = img.dtype
    krnl = np.array([[0,  0, 0,  0, 1,   0, 1,  0, 0,  0, 0],
                     [0,  0, 0, -1, 0,  -2, 0, -1, 0,  0, 0],
                     [0,  0, 1,  1, 2,   1, 2,  1, 1,  0, 0],
                     [0, -1, 1, -5, 3,  -9, 3, -5, 1, -1, 0],
                     [1,  0, 2,  3, 1,   7, 1,  3, 2,  0, 1],
                     [0, -2, 1, -9, 7, 104, 7, -9, 1, -2, 0],
                     [1,  0, 2,  3, 1,   7, 1,  3, 2,  0, 1],
                     [0, -1, 1, -5, 3,  -9, 3, -5, 1, -1, 0],
                     [0,  0, 1,  1, 2,   1, 2,  1, 1,  0, 0],
                     [0,  0, 0, -1, 0,  -2, 0, -1, 0,  0, 0],
                     [0,  0, 0,  0, 1,   0, 1,  0, 0,  0, 0]], dtype=dtype)
    krnl /= 128.
    krnl = krnl.astype(dtype)
    return spimg.convolve(img, krnl, mode='mirror')


# linear demosaicing based on luminance and chrominance splitting
def linear_demosaic(image, lumfilter='kernel', interp='bilinear', **kwargs):
    # estimate the luminance
    imagef = cfa.rgbmosaic2_ch1(image)
    if lumfilter == 'kernel':
        luminance = _luminance_filter(imagef)
    else:
        luminance = fft_filter(imagef, *lumfilter)

    # estimate the chrominance
    chrominance = cfa.ch1mosaic2_rgb(imagef - luminance)
    if interp == 'bilinear':
        chrominance = bilinear_interpolation(chrominance)
    elif interp == 'adaptive':
        chrominance = adaptive_interpolation(chrominance, **kwargs)
    elif interp == 'adaptive_lum':
        chrominance = adaptive_interpolation(chrominance, classifier='grad',
                                             img_class=luminance, **kwargs)

    # compose the image
    image_out = np.zeros_like(chrominance)
    image_out[:, :, 0] = luminance
    image_out[:, :, 1] = luminance
    image_out[:, :, 2] = luminance
    image_out += chrominance

    # clip the values
    if np.issubdtype(image.dtype, np.integer):
        iinfo = np.iinfo(image.dtype)
        vmin = iinfo.min
        vmax = iinfo.max
    else:
        vmin = 0.
        vmax = 1.
    np.clip(image_out, vmin, vmax, out=image_out)

    return image_out.astype(image.dtype)
