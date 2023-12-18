from string import ascii_lowercase
from random import choice, randint
import os
import argparse
from cfa_image import pics_in_dir

if __name__ == '__main__':
    desc = 'Rename all names in a random way'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('directory', metavar='dirin', type=str,
                        help='Path to the directory containing the images')
    args = parser.parse_args()

    imglist = pics_in_dir(args.directory)
    nimgs = len(imglist)
    for ik, f in enumerate(imglist):
        if (ik % 500) == 0:
            print('Proccessed {:d} out of {:d}'.format(ik, nimgs))
        if not os.path.isfile(f):
            continue
        newpath = os.path.join(args.directory, ''.join([choice(ascii_lowercase) for _ in range(randint(10, 25))]))
        #print("rename {} to {}".format(f, newpath))
        os.rename(f, newpath + ".png")
