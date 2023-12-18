import os
import re
import numpy as np
import cv2


def frame_from_dir(dirpics, ending='', frames=None):
    '''Try to extract the files that make up a single frame.

    Frames are cut into smaller patches, which are saved with the filename
    ``NAMEX-CCCC-patch-YI-YJ_ending.png``, where ``NAME`` is the original
    picture name, ``X`` is the frame number, ``CCCC`` is the CFA identifier
    (this part is completly ignored) and ``YI`` and ``YJ`` are the patch
    offsets in the x and y direction, when placed in the frame.

    Parameters
    ----------
    dirpics : str
        Directory from where to extract the frames files.
    ending : str, optional
        The ending on the filename. Files without that ending will be
        discarded. By default an empty string.
    frames : list, optional
        Start of filenames of interest. If None, all frames found will be
        included in the list of files. By default None.

    Returns
    -------
    list
        List containing a list of filenames for each frame.
    '''
    # first pass, filter out irrelevant files
    flist = os.listdir(dirpics)
    flist = filter(lambda x: x.endswith('.png') and ('-patch-' in x)
                   and (ending in x), flist)
    flist = list(flist)
    flist.sort()

    # number of frames present
    frame_list = map(lambda x: re.findall(r'^[a-zA-Z]+\d+', x), flist)
    frame_list = [fl[0] for fl in frame_list if fl]
    frame_list = np.unique(list(frame_list))
    if frames is not None:
        frame_list = np.intersect1d(frame_list, frames)

    if frame_list.size == 0:
        return []

    # split the files in each frame
    frame_files = []
    for fname in frame_list:
        tmpl = [fl for fl in flist if fl.startswith(fname + '-')]
        frame_files.append([os.path.join(dirpics, fl) for fl in tmpl])

    return frame_files


def stitch_frame(frame_file, dirout):
    '''Stitch together the provided patches to a frame.

    Given a list of files belonging to the same frame, the stitched image is
    saved in the ``dirout`` path.

    Parameters
    ----------
    frame_file : list
        List of the files that make up the frame.
    dirout : str
        Path where to export the reconstructed frame.

    Raises
    ------
    RuntimeError
        If the size of the patches do not fit together.
    '''
    img_list = []
    ijlist = []
    img_shape = []
    for fl in frame_file:
        img_list.append(cv2.imread(fl, -cv2.IMREAD_ANYDEPTH))
        ipatch = re.findall(r'(?<=patch-)\d+-\d+', fl)[0]
        ipatch = ipatch.split('-')
        ijlist.append([int(ipatch[0]), int(ipatch[1])])
        img_shape.append(img_list[-1].shape)

    ijlist = np.asarray(ijlist)
    shape = np.amax(ijlist, axis=0) + 1
    inds = np.lexsort((ijlist[:, 1], ijlist[:, 0]))
    tot_ishape = np.zeros(shape, dtype=int)
    tot_jshape = np.zeros(shape, dtype=int)
    for ik in inds:
        ii, ij = ijlist[ik]
        tot_ishape[ii, ij] = img_shape[ik][0]
        tot_jshape[ii, ij] = img_shape[ik][1]

    if np.diff(tot_ishape, axis=1).any() or np.diff(tot_jshape, axis=0).any():
        raise RuntimeError('Patches do not have equal sizes')

    tot_ishape = tot_ishape[:, 0]
    tot_jshape = tot_jshape[0, :]

    mind = np.ravel_multi_index(ijlist.T, shape)
    img_list = [img_list[ik] for ik in np.argsort(mind)]
    img_stitched = []
    for ij in range(shape[0]):
        idxs = ij * shape[1]
        idxf = idxs + shape[1]
        img_stitched.append(cv2.hconcat(img_list[idxs:idxf]))
    img_stitched = cv2.vconcat(img_stitched)

    fname0 = os.path.split(frame_file[0])[1]
    fname0, fext = os.path.splitext(fname0)
    frmname = re.findall(r'^[a-zA-Z]+\d+', fname0)[0]
    filesave = os.path.join(dirout, frmname + fext)
    cv2.imwrite(filesave, img_stitched)


def stitch_list(imglist, filename=None):
    '''Stitch image from list of patches.

    Parameters
    ----------
    imglist : list
        List of list of images, outer list containes the rows and inner list
        the images per row
    filename : str, optional
       If not None save the image to this path, by default None

    Returns
    -------
    array_like
        The stitched image
    '''
    img_stitched = []
    for row in imglist:
        img_stitched.append(cv2.hconcat(row))
    img_stitched = cv2.vconcat(img_stitched)

    if filename is not None:
        cv2.imwrite(filename, cv2.cvtColor(img_stitched, cv2.COLOR_RGB2BGR))
    return img_stitched


def cut_frame(picname, shape=128, offset=0):
    '''Cut a frame into patches.

    Parameters
    ----------
    picname : str
        Path to the input image.
    shape : int, list, optional
        Shape of the output patches. A list of two component is expected, for
        the height and the width of the patch. If the input is an integer, then
        the same value will be used in both dimensions. By default 128.
    offset : int, list, optional
        Offset for cutting the patches. The offset is measured from the top
        left corner of the image. A list of two component is expected. If the
        input is an integer, then the same value will be used in both
        dimensions. By default 128. By default 0.

    Returns
    -------
    list
        List of lists of images. Outer list contains rows and the inner list
        the patches on each row.

    Raises
    ------
    RuntimeError
        If the patch size with the offset is smaller than original image size.
    '''
    # check the input arguments
    if isinstance(shape, int):
        shp_cut = [shape, ] * 2
    else:
        shp_cut = list(shape)
    if isinstance(offset, int):
        offst = [offset, ] * 2
    else:
        offst = list(offset)

    # load and check shape
    img = cv2.imread(picname, -cv2.IMREAD_ANYDEPTH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


def save_patches(outdir, imglist, imgname, starting='', ending=''):
    '''Save a list of patches into a foler

    Parameters
    ----------
    outdir : str
        Folder path to output the patches.
    imglist : list
        List of list containing the patches in rows.
    imgname : str
        Path of the original image
    starting : str, optional
        String to append at the start of the patch filenames, by default ''.
    ending : str, optional
        String to append at the end of the patch filenames, by default ''.
    '''
    fname = os.path.split(imgname)[1]
    fname, fext = os.path.splitext(fname)
    fname = starting + fname + '-patch-{:d}-{:d}' + ending + fext
    save_file = os.path.join(outdir, fname)
    for ij, img_row in enumerate(imglist):
        for ii, img in enumerate(img_row):
            cv2.imwrite(save_file.format(ij, ii),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
