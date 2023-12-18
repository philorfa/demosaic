import argparse
import sys
from video_library import data_from_videos, copy_directories

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument('-FolderIn', '--FolderIn', nargs="+", type=str,
                   help='Name(s) of the folder containing the videos')

    a.add_argument('-FolderMerg', '--FolderMerg', nargs="+", type=str,
                   help='Name of the folder to be merged', default=None)

    a.add_argument('-FolderOut', '--FolderOut', nargs="+", type=str,
                   help='Name of the output folders')

    a.add_argument('-cfa', '--color_filter', action='store', nargs='+',
                   type=str, help='Color filter', default="chame")

    a.add_argument('-s', '--shape', action='store', nargs='+',
                   type=int, help='Patch size', default=[128, 128])

    a.add_argument('-o', '--offset', action='store', nargs='+',
                   type=int, help='Offset from the top left corner',
                   default=[0, 0])

    a.add_argument('-ord', '--order', type=str, action='store', nargs='?',
                   help='Pixel order [RGGB, BRRG, GRBG, GBRG]',
                   default='GRBG')

    args = a.parse_args()

    print("> Extracting Data from", args.FolderIn[0], "!!!\n")
    
    root_dir = "..\\data_"+ args.order.lower()

    data_from_videos(args.FolderIn, args.FolderOut, args.color_filter, args.shape, args.offset, args.order)

    
    
    if args.FolderMerg[0] is not None:
        
        print("> Merging ...")
        #  Train Directories
        
        train_rgb_src = root_dir + "\\train\\" + args.FolderMerg[0] + str(args.shape[0]) + "\\dataset-" + args.FolderMerg[0]
        train_rgb_dest = root_dir + "\\train\\" + args.FolderOut[0] + str(args.shape[0]) + "\\dataset-" + args.FolderOut[0]

        copy_directories(train_rgb_src, train_rgb_dest)

        # 1channel Mosaiced
        train_mos_1ch_src = root_dir + "\\train\\" + args.FolderMerg[0] + str(args.shape[0]) + \
                            "\\data-1ch\\dataset-" + args.FolderMerg[0] + "-" + args.color_filter

        train_mos_1ch_dest = root_dir + "\\train\\" + args.FolderOut[0] + str(args.shape[0]) + \
                             "\\data-1ch\\dataset-" + args.FolderOut[0] + "-" + args.color_filter

        copy_directories(train_mos_1ch_src, train_mos_1ch_dest)

        # 3channel Mosaiced
        train_mos_3ch_src = root_dir + "\\train\\" + args.FolderMerg[0] + str(args.shape[0]) + \
                            "\\data-3ch\\dataset-" + args.FolderMerg[0] + "-" + args.color_filter

        train_mos_3ch_dest = root_dir + "\\train\\" + args.FolderOut[0] + str(args.shape[0]) + \
                             "\\data-3ch\\dataset-" + args.FolderOut[0] + "-" + args.color_filter

        copy_directories(train_mos_3ch_src, train_mos_3ch_dest)

        #  Test Directories
        test_rgb_src = root_dir + "\\test\\" + args.FolderMerg[0] + str(args.shape[0]) + "\\dataset-" + args.FolderMerg[0]
        test_rgb_dest = root_dir + "\\test\\" + args.FolderOut[0] + str(args.shape[0]) + "\\dataset-" + args.FolderOut[0]

        copy_directories(test_rgb_src, test_rgb_dest)

        # 1channel Mosaiced
        test_mos_1ch_src = root_dir + "\\test\\" + args.FolderMerg[0] + str(args.shape[0]) + \
                           "\\data-1ch\\dataset-" + args.FolderMerg[0] + "-" + args.color_filter

        test_mos_1ch_dest = root_dir + "\\test\\" + args.FolderOut[0] + str(args.shape[0]) + \
                            "\\data-1ch\\dataset-" + args.FolderOut[0] + "-" + args.color_filter

        copy_directories(test_mos_1ch_src, test_mos_1ch_dest)

        # 3channel Mosaiced
        test_mos_3ch_src = root_dir + "\\test\\" + args.FolderMerg[0] + str(args.shape[0]) + \
                           "\\data-3ch\\dataset-" + args.FolderMerg[0] + "-" + args.color_filter

        test_mos_3ch_dest = root_dir + "\\test\\" + args.FolderOut[0] + str(args.shape[0]) + \
                            "\\data-3ch\\dataset-" + args.FolderOut[0] + "-" + args.color_filter

        copy_directories(test_mos_3ch_src, test_mos_3ch_dest)

    sys.exit(0)
