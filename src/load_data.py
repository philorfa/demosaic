import numpy as np
import cv2
import os

path = "../data/dataset-cfa-rbg/output"
save_path = "../data/dataset-cfa-rbg"
file_list = os.listdir(path)
sorted(file_list)

for fname in file_list:
    try:
        img = np.fromfile(os.path.join(path, fname), dtype='uint16')
        img.shape
        h = 128
        w = 128

        factor = 64

        if fname.endswith(".rgb"):
            print(fname, " -> rgb to png")
            if img.shape[0] == w * h * 3:
                img = img * factor
                cv2.imwrite(os.path.join(save_path, "{}_rgb.png".format(fname[:-4])),
                            cv2.cvtColor(img.reshape(h, w, 3), cv2.COLOR_RGB2BGR))
            elif img.shape[0] != w * h:
                print("size mismatch")


    except Exception as e:
        print(e)
