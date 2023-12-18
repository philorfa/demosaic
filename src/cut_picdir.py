#!/usr/bin/env python3
from cut_stitch_pics import cut_frame, save_patches
from cfa_image import pics_in_dir
import argparse
import os
import sys


if __name__ == '__main__':
    desc = 'Cut all picture found in directory into patches.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('directory', metavar='dirin', type=str,
                        help='Path to the directory containing the images')
    parser.add_argument('dirpath', metavar='dirout', type=str,
                        help='Path where to export the patches')
    parser.add_argument('-s', '--shape', action='store', nargs='+',
                        type=int, help='Patch size', default=[128, 128])
    parser.add_argument('-o', '--offset', action='store', nargs='+',
                        type=int, help='Offset from the top left corner',
                        default=[0, 0])
    parser.add_argument('-e', '--ending', action='store', nargs='?',
                        default='', type=str, const='',
                        help='Append this string at the end of filenames')
    parser.add_argument('-t', '--starting', action='store', nargs='?',
                        default='', type=str, const='',
                        help='Append this string at the start of filenames')
    args = parser.parse_args()

    if not os.path.exists(args.dirpath):
        os.makedirs(args.dirpath)

    shape = args.shape
    if len(shape) == 1:
        shape = shape[0]
    offset = args.offset
    if len(offset) == 1:
        offset = offset[0]

    imglist = pics_in_dir(args.directory)
    nimgs = len(imglist)
    for ik, img in enumerate(imglist):
        if (ik % 500) == 0:
            print('Proccessed {:d} out of {:d}'.format(ik, nimgs))
        img_cut = cut_frame(img, shape, offset)
        save_patches(args.dirpath, img_cut, img, args.starting, args.ending)
    sys.exit(0)
