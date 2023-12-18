import argparse
import sys
from video_library import data_from_videos


if __name__ == "__main__":
    a = argparse.ArgumentParser()

    a.add_argument('-FolderIn', '--FolderIn', nargs="+", type=str,
                   help='Name(s) of the folder containing the videos')

    a.add_argument('-FolderOut', '--FolderOut', nargs="+", type=str,
                   help='Name(s) of the output folders')

    a.add_argument('-cfa', '--color_filter', action='store', nargs='+',
                   type=str, help='Color filter', default="chame")

    a.add_argument('-s', '--shape', action='store', nargs='+',
                   type=int, help='Patch size', default=[128, 128])

    a.add_argument('-o', '--offset', action='store', nargs='+',
                   type=int, help='Offset from the top left corner',
                   default=[0, 0])

    a.add_argument('-ord', '--order', type=str, action='store', nargs='?',
                   help='Pixel order [RGGB, BRRG, GRBG, GBRG]',
                   default='RGGB', const='RGGB')

    args = a.parse_args()

    data_from_videos(args.FolderIn, args.FolderOut, args.color_filter, args.shape, args.offset, args.order)

    sys.exit(0)
