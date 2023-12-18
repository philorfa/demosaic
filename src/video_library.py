from pathlib import Path
import random
from os import path
import cv2
import os
import secrets
import string
import shutil
import numpy as np
from cfa_image import mosaic
import math

try:
    from tqdm import tqdm

    use_tqdm = True
except ImportError:
    use_tqdm = False


def cut_frame(img, shape=128, offset=0):
    # check the input arguments
    if isinstance(shape, int):
        shp_cut = [shape, ] * 2
    else:
        shp_cut = list(shape)
    if isinstance(offset, int):
        offst = [offset, ] * 2
    else:
        offst = list(offset)

    shp_orig = img.shape
    if shp_cut[0] + offst[0] > shp_orig[0]:
        raise RuntimeError('Patch does not fit in height')
    if shp_cut[1] + offst[1] > shp_orig[1]:
        raise RuntimeError('Patch does not fit in width')

    # do the cut
    xcut = np.arange(offst[1], shp_orig[1] + 1, shp_cut[1])[:-1]
    ycut = np.arange(offst[0], shp_orig[0] + 1, shp_cut[0])[:-1]

    img_cut = []
    for iy in ycut:
        row_cut = []
        for ix in xcut:
            imcut = img[iy:iy + shp_cut[0], ix:ix + shp_cut[1]]
            row_cut.append(imcut)
        img_cut.append(row_cut)
    return img_cut


def save_patches(outdir, imglist):
    abc = string.ascii_letters
    digits = string.digits

    for ij, img_row in enumerate(imglist):
        for ii, img in enumerate(img_row):
            fname = "".join(secrets.choice(digits + abc) for _ in range(15))
            save_file = os.path.join(outdir, fname + ".png")
            cv2.imwrite(save_file, img)


def videos_in_dir(directory):
    vid_ext = '.mp4'
    flist = os.listdir(directory)
    flist = filter(lambda x: x.endswith(vid_ext), flist)
    return [os.path.join(directory, fl) for fl in flist]


def extractImages(pathIn, pathOut, shape_patch, offset_patch):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        if success:
            img_cut = cut_frame(image, shape_patch, offset_patch)
            save_patches(pathOut, img_cut)
            count = count + 1


def data_from_videos(FolderIn, FolderOut, color_filter, shape, offset, order):
    if len(FolderOut) == 1 and len(FolderIn) != 1:
        print("Merging all Folders of Videos into", FolderOut[0])
        mode = 1
    elif len(FolderOut) == len(FolderIn):
        print("1-1 Dataset Creation")
        mode = 2
    else:
        raise ValueError("Folders input-output should be matching")

    shape = shape[0]
    # if len(shape) == 1:
    #     shape = shape[0]
    offset = offset[0]
    # if len(offset) == 1:
    #     offset = offset[0]

    root_dir = "..\\data_" + order.lower()
    folder_directory = root_dir + "\\Videos\\"
    train_save_directory = root_dir + "\\train\\"
    test_save_directory = root_dir + "\\test\\"

    if not os.path.exists(train_save_directory):
        os.makedirs(train_save_directory)

    if not os.path.exists(test_save_directory):
        os.makedirs(test_save_directory)

    for folder in FolderOut:

        if os.path.exists(os.path.join(root_dir, folder)):
            shutil.rmtree(os.path.join(root_dir, folder))

        rgb_folder = os.path.join(root_dir, folder, folder + str(shape))
        os.makedirs(rgb_folder)

        fld_train = os.path.join(train_save_directory, folder + str(shape))
        if os.path.exists(fld_train):
            shutil.rmtree(fld_train)

        os.makedirs(fld_train)

        rgb_train = os.path.join(fld_train, "dataset-" + folder)

        single_channel_train = os.path.join(fld_train, "data-1ch")
        single_channel_train_cfa = os.path.join(single_channel_train,
                                                "dataset-" + folder + "-" + color_filter)

        three_channel_train = os.path.join(fld_train, "data-3ch")
        three_channel_train_cfa = os.path.join(three_channel_train,
                                               "dataset-" + folder + "-" + color_filter)

        os.makedirs(rgb_train)
        os.makedirs(single_channel_train_cfa)
        os.makedirs(three_channel_train_cfa)

        fld_test = os.path.join(test_save_directory, folder + str(shape))
        if os.path.exists(fld_test):
            shutil.rmtree(fld_test)

        os.makedirs(fld_test)

        rgb_test = os.path.join(fld_test, "dataset-" + folder)

        single_channel_test = os.path.join(fld_test, "data-1ch")
        single_channel_test_cfa = os.path.join(single_channel_test,
                                               "dataset-" + folder + "-" + color_filter)

        three_channel_test = os.path.join(fld_test, "data-3ch")
        three_channel_test_cfa = os.path.join(three_channel_test,
                                              "dataset-" + folder + "-" + color_filter)

        os.makedirs(rgb_test)
        os.makedirs(single_channel_test_cfa)
        os.makedirs(three_channel_test_cfa)

    for jk, folder in enumerate(FolderIn):
        folder_dir = os.path.join(folder_directory, folder)
        if mode == 1:
            folder_output = os.path.join(root_dir, FolderOut[0],
                                         FolderOut[0] + str(shape))
        else:
            folder_output = os.path.join(root_dir, FolderOut[jk],
                                         FolderOut[jk] + str(shape))
        print("> Extracting patches from ", folder_dir)
        vidlist = videos_in_dir(folder_dir)
        nvids = len(vidlist)

        for ik, video in enumerate(vidlist):
            if ((ik + 1) % 1) == 0:
                print(
                    '> Create image patches from video {:d} out of {:d} ...'.format(
                        ik + 1, nvids))
            extractImages(video, folder_output, shape, offset)
        print("> Saved them to", folder_output)

    train = 0.9
    test = 1 - train
    print("> Split train-test({:f},{:f}) set ...".format(train, test))

    for folder in FolderOut:
        input = os.path.join(root_dir, folder)
        output = folder
        ratio(input, order, output=output, shape=shape, seed=1337,
              ratio=(train, 0, test))
        shutil.rmtree(input)

    split = ["train", "test"]
    one_channel = ["1", "3"]

    # 1. Train-3ch
    # 2. Train-1ch
    # 3. Test-3ch
    # 4. Test-1ch
    for folder in FolderOut:
        for split_dataset in split:
            for channel in one_channel:

                input_directory = os.path.join(root_dir, split_dataset,
                                               folder + str(shape),
                                               "dataset-" + folder)
                output_directory = os.path.join(root_dir, split_dataset,
                                                folder + str(shape),
                                                "data-" + channel + "ch",
                                                "dataset-" + folder + "-" + color_filter)

                flist = os.listdir(input_directory)
                flist = filter(lambda x: x.endswith('.png'), flist)
                flist = list(flist)
                flist = [os.path.join(input_directory, fl) for fl in flist]

                num_files = len(flist)

                print("> Saving Mosaiced patches on ", output_directory, " ...")

                for ij, fl in enumerate(flist):
                    img = cv2.imread(fl, -cv2.IMREAD_ANYDEPTH)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    dtype = img.dtype
                    fname = os.path.split(fl)[1]
                    fname, fext = os.path.splitext(fname)
                    fname = os.path.join(output_directory, fname + ".png")
                    img = mosaic(img, color_filter, order, channel)
                    cv2.imwrite(fname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if ij % (math.ceil(num_files / 10)) == 0:
                        print(ij + 1, 'out of', num_files)


def list_dirs(directory):
    """
    Returns all directories in a given directory
    """
    return [f for f in Path(directory).iterdir() if f.is_dir()]


def list_files(directory):
    """
    Returns all files in a given directory
    """
    return [
        f
        for f in Path(directory).iterdir()
        if f.is_file() and not f.name.startswith(".")
    ]


def check_input_format(input):
    p_input = Path(input)
    if not p_input.exists():
        err_msg = f'The provided input folder "{input}" does not exists.'
        if not p_input.is_absolute():
            err_msg += f' Your relative path cannot be found from the current working directory "{Path.cwd()}".'
        raise ValueError(err_msg)

    if not p_input.is_dir():
        raise ValueError(
            f'The provided input folder "{input}" is not a directory')

    dirs = list_dirs(input)
    if len(dirs) == 0:
        raise ValueError(
            f'The input data is not in a right format. Within your folder "{input}" there are no directories. Consult the documentation how to the folder structure should look like.'
        )


def ratio(
        input,
        order,
        output="output",
        shape=128,
        seed=1337,
        ratio=(0.8, 0.1, 0.1),
        group_prefix=None,
        move=False,
):
    if not round(sum(ratio), 5) == 1:  # round for floating imprecision
        raise ValueError("The sums of `ratio` is over 1.")
    if not len(ratio) in (2, 3):
        raise ValueError("`ratio` should")

    check_input_format(input)

    if use_tqdm:
        prog_bar = tqdm(desc=f"Copying files", unit=" files")

    for class_dir in list_dirs(input):
        split_class_dir_ratio(
            class_dir,
            order,
            output,
            shape,
            ratio,
            seed,
            prog_bar if use_tqdm else None,
            group_prefix,
            move,
        )

    if use_tqdm:
        prog_bar.close()


def group_by_prefix(files, len_pairs):
    """
    Split files into groups of len `len_pairs` based on their prefix.
    """
    results = []
    results_set = set()  # for fast lookup, only file names
    for f in files:
        if f.name in results_set:
            continue
        f_sub = f.name
        for _ in range(len(f_sub)):
            matches = [
                x
                for x in files
                if x.name not in results_set
                   and x.name.startswith(f_sub)
                   and f.name != x.name
            ]
            if len(matches) == len_pairs - 1:
                results.append((f, *matches))
                results_set.update((f.name, *[x.name for x in matches]))
                break
            elif len(matches) < len_pairs - 1:
                f_sub = f_sub[:-1]
            else:
                raise ValueError(
                    f"The length of pairs has to be equal. Coudn't find {len_pairs - 1} matches for {f}. Found {len(matches)} matches."
                )
        else:
            raise ValueError(f"No adequate matches found for {f}.")

    if len(results_set) != len(files):
        raise ValueError(
            f"Could not find enough matches ({len(results_set)}) for all files ({len(files)})"
        )
    return results


def setup_files(class_dir, seed, group_prefix=None):
    """
    Returns shuffled list of filenames
    """
    random.seed(seed)  # make sure its reproducible

    files = list_files(class_dir)

    if group_prefix is not None:
        files = group_by_prefix(files, group_prefix)

    files.sort()
    random.shuffle(files)
    return files


def split_class_dir_ratio(class_dir, order, output, shape, ratio, seed,
                          prog_bar, group_prefix, move):
    """
    Splits a class folder
    """
    files = setup_files(class_dir, seed, group_prefix)

    # the data was shuffled already
    split_train_idx = int(ratio[0] * len(files))
    split_val_idx = split_train_idx + int(ratio[1] * len(files))

    li = split_files(files, split_train_idx, split_val_idx, len(ratio) == 3)
    copy_files(li, order, class_dir, output, shape, prog_bar, move)


def split_files(files, split_train_idx, split_val_idx, use_test, max_test=None):
    """
    Splits the files along the provided indices
    """
    files_train = files[:split_train_idx]
    files_val = (
        files[split_train_idx:split_val_idx] if use_test else files[
                                                              split_train_idx:]
    )

    li = [(files_train, "train"), (files_val, "val")]

    # optional test folder
    if use_test:
        files_test = files[split_val_idx:]
        if max_test is not None:
            files_test = files_test[:max_test]

        li.append((files_test, "test"))
    return li


def copy_files(files_type, order, class_dir, output, shape, prog_bar, move):
    """
    Copies the files from the input folder to the output folder
    """

    copy_fun = shutil.move if move else shutil.copy2

    # get the last part within the file
    # class_name = path.split(class_dir)[1]
    for (files, folder_type) in files_type:
        if folder_type == "val":
            continue

        root_dir = "..\\data_" + order.lower()
        full_path = path.join(root_dir, folder_type, output + str(shape),
                              "dataset-" + output)
        Path(full_path).mkdir(parents=True, exist_ok=True)
        for f in files:
            if not prog_bar is None:
                prog_bar.update()
            if type(f) == tuple:
                for x in f:
                    copy_fun(str(x), str(full_path))
            else:
                copy_fun(str(f), str(full_path))


def copy_directories(src_directory, dst_directory):
    # Extract file from src directory then create a dst directory and copy into dst directory
    print("> ", src_directory, " --> ", dst_directory)
    flist = os.listdir(src_directory)
    num_files = len(flist)
    for ik, file in enumerate(os.listdir(src_directory)):
        src_file = os.path.join(src_directory, file)
        dst_file = os.path.join(dst_directory, file)

        if os.path.exists(dst_file):
            # in case of the src and dst are the same file
            if os.path.samefile(src_file, dst_file):
                continue
            os.remove(dst_file)
        shutil.copy2(src_file, dst_file)
        if (ik + 1) % (math.ceil(num_files / 10)) == 0:
            print(ik + 1, 'out of', num_files)
