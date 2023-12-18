#!/usr/bin/env python3
from cut_stitch_pics import frame_from_dir, stitch_frame
import argparse
import sys
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stitch frame from patches.')
    parser.add_argument('dirpath', metavar='dirin', type=str,
                        help='Path where the patches are stored')
    parser.add_argument('outpath', metavar='dirout', type=str,
                        help='Path where the output images are saved')
    parser.add_argument('-e', '--ending', action='store', nargs='?',
                        default='', type=str, const='',
                        help='Consider only files containing this sting')
    parser.add_argument('-f', '--frames', action='store', nargs='*',
                        type=str, help='List of frames of interest')
    args = parser.parse_args()

    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    frame_list = frame_from_dir(args.dirpath, args.ending, args.frames)
    if not frame_list:
        print('No frame patches found in', args.dirpath)
        sys.exit(1)
    for frm in frame_list:
        stitch_frame(frm, args.outpath)

    sys.exit(0)
