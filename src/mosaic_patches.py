#!/usr/bin/env python3
from cfa_image import mosaic
import os
import sys
import argparse
import cv2
import math


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply CFA on images.')
    parser.add_argument('img', metavar='image', type=str,
                        help='Image or directory of images as input')
    parser.add_argument('dirpath', metavar='dirout', type=str,
                        help='Path where to export the patches')
    parser.add_argument('-c', '--cfa', type=str, action='store', nargs='?',
                        help='CFA to apply [bayer, tetra, nona, chame]',
                        default='bayer', const='bayer')
    parser.add_argument('-o', '--order', type=str, action='store', nargs='?',
                        help='Pixel order [RGGB, BRRG, GRBG, GBRG]',
                        default='RGGB', const='RGGB')
    parser.add_argument('-1', '--one_channel', action='store_true',
                        help='Export CFA image as single channel')
    parser.add_argument('-e', '--ending', action='store', nargs='?',
                        default='', type=str, const='',
                        help='Append this sting at the end of filenames')
    args = parser.parse_args()

    # find files / folder
    if os.path.isfile(args.img):
        flist = [args.img, ]
    elif os.path.isdir(args.img):
        flist = os.listdir(args.img)
        flist = filter(lambda x: x.endswith('.png'), flist)
        flist = list(flist)
        flist = [os.path.join(args.img, fl) for fl in flist]
    else:
        raise FileNotFoundError('Could find file or folder', args.img)

    if not os.path.exists(args.dirpath):
        os.makedirs(args.dirpath)
        
    num_files = len(flist)
    for ik, fl in enumerate(flist):
        img = cv2.imread(fl, -cv2.IMREAD_ANYDEPTH)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dtype = img.dtype
        fname = os.path.split(fl)[1]
        fname, fext = os.path.splitext(fname)
        fname += args.ending + fext
        fname = os.path.join(args.dirpath, fname)
        img = mosaic(img, args.cfa, args.order, args.one_channel)
        cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if ik % (math.ceil(num_files / 100)) == 0:
            print(ik+1, 'out of', num_files)
    sys.exit(0)
