#!/usr/bin/env python3
from cut_stitch_pics import cut_frame, save_patches
import argparse
import os
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cut frame into patches.')
    parser.add_argument('img', metavar='image', type=str,
                        help='Image to cut')
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

    img_cut = cut_frame(args.img, shape, offset)
    save_patches(args.dirpath, img_cut, args.img, args.starting, args.ending)
    sys.exit(0)
