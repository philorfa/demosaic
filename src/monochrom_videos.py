import cv2
import numpy as np
import os
import argparse
import random
import string
import secrets
from tqdm import tqdm

if __name__ == "__main__":
    a = argparse.ArgumentParser()

    a.add_argument('-FolderOut', '--FolderOut', type=str,
                   help='Name of the output folder')

    a.add_argument('-vid', '--videos', type=int,
                   help='Number of videos to produce', default=1)

    a.add_argument('-w', '--width', type=int,
                   help='Frame width', default=1920)

    a.add_argument('-ht', '--height', type=int,
                   help='Frame height', default=1080)

    a.add_argument('-d', '--duration', type=int,
                   help='Duration of video(s) in sec', default=10)

    a.add_argument('-fps', '--fps', type=int,
                   help='Frames per second', default=1)

    a.add_argument('-sq', '--square_size', type=int,
                   help='Square Size', default=45)

    args = a.parse_args()

    for i in range(args.videos):

        print("> Creating Video #", i + 1, " out of ", args.videos)
        total_frames = args.duration * args.fps

        # Create the video writer object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        abc = string.ascii_letters
        digits = string.digits
        vname = "".join(secrets.choice(digits + abc) for _ in range(7))

        outdir = os.path.join("..\\data\\Videos", args.FolderOut)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        output_path = os.path.join(outdir, vname + ".mp4")
        out = cv2.VideoWriter(output_path, fourcc, args.fps, (args.width, args.height))

        for k in range(total_frames):

            mode = np.random.randint(1, 3)
            frame = np.zeros((args.height, args.width, 3), np.uint8)

            # red, green, yellow
            color = [(0, 0, 255), (0, 255, 0), (0, 255, 255)]

            color_range = 50

            color_frame = random.choice(color)
            # Generate random variations of the color
            bgr = np.array(color_frame, dtype=np.int32)
            bgr += np.random.randint(-color_range, color_range + 1, size=3)
            bgr = np.clip(bgr, 0, 255)
            # Apply the color to the corresponding pixel of the image
            bgr = bgr.astype(np.uint8)

            # Create a numpy array with the specified color
            frame[:] = bgr

            if mode == 2:

                for i in range(0,args.height-args.square_size, args.square_size):

                    for j in range(0,args.width-args.square_size, args.square_size):

                        square = np.zeros((args.square_size, args.square_size, 3), np.uint8)

                        color_square = random.choice(color)
                        # Generate random variations of the color
                        bgr = np.array(color_square, dtype=np.int32)
                        bgr += np.random.randint(-color_range, color_range + 1, size=3)
                        bgr = np.clip(bgr, 0, 255)
                        # Apply the color to the corresponding pixel of the image
                        bgr = bgr.astype(np.uint8)

                        square[:] = bgr
                        frame[i:i + args.square_size, j:j + args.square_size, :] = square

            # Write the frames to the video
            out.write(frame)

        # Release the video writer object
        out.release()

        # Print the path to the output video file
        print(f"Video saved to {os.path.abspath(output_path)}")
