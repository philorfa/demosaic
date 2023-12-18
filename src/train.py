from argparse import ArgumentParser
from model import PrunablePReLu, PrunableConcat
from model import PrunableDecompose1D, PrunableClip
from train_test_model import training_session, training_session_distribute
from keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import json

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu_avail = tf.config.list_physical_devices('GPU')
ngpu = len(gpu_avail)
for igpu in gpu_avail:
    print(igpu)

if __name__ == '__main__':

    parser = ArgumentParser(add_help=False)
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('-mn', '--model_number', type=int, required=True)
    optional.add_argument('-ds', '--dataset', default='medium',
                          type=str, nargs='+',
                          help='Dataset or datasets to be used for training')
    optional.add_argument('-e', '--epochs', default=5, type=int)
    optional.add_argument('-bs', '--batch_size', default=16, type=int)
    optional.add_argument('-ls', '--learning_rate_sch', default=0, type=int)
    optional.add_argument('-ct', '--continue_train', nargs='?', default=None,
                          const=-1., type=float,
                          help='Continue training from stored checkpoint')
    optional.add_argument('-ng', '--gpu-list', nargs='+', type=int,
                          help='Select the GPUs you want to use')
    optional.add_argument('-tm', '--train-metric', default='l2', type=str,
                          help='l2 or cpsnr or ssim or'
                               ' ssim_l1_mix or cpsnr_l1_mix')
    optional.add_argument('-ord', '--order_pattern', default='grbg', type=str,
                          help='order pattern')
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        help='show this help message and exit'
    )

    args = parser.parse_args()

    model_dir = os.path.join('..', 'saved_models',
                             'model' + str(args.model_number))

    with open(os.path.join(model_dir, 'args_create.txt'), 'r') as f:
        args.__dict__.update(json.load(f))

    if args.color_filter not in ['bayer', 'nona', 'tetra', 'chame']:
        raise ValueError("CFA is not compatible")
    if args.learning_rate_sch not in [0, 1, 2]:
        raise ValueError("Learning rate Scheduler is not compatible")
    distribute = False
    if args.gpu_list is not None:
        glist = np.unique(args.gpu_list)
        if np.any(glist < 0):
            raise ValueError('All GPU ids should be positive')
        if (np.max(glist) > (ngpu - 1)) or (len(glist) > ngpu):
            raise RuntimeError('Cannot find requested GPUs')
        if len(glist) == 1:
            dev_string = '/device:GPU:{:d}'.format(glist[0])
            dev = tf.device(dev_string)
        else:
            gpu_log = tf.config.list_logical_devices('GPU')
            gpu2use = []
            dev_string = ''
            for ik in glist:
                gpu2use.append(gpu_log[ik])
                dev_string += ', ' + gpu_log[ik].name
            # Alternative strategies
            # strategy = tf.distribute.MirroredStrategy(
            #     devices=gpu2use,
            #     cross_device_ops=tf.distribute.ReductionToOneDevice()
            # )
            # strategy = tf.distribute.MirroredStrategy(
            #     devices=gpu2use,
            #     cross_device_ops=tf.distribute.NcclAllReduce()
            # )
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
            dev = strategy.scope()
            distribute = True
    else:
        if ngpu > 0:
            dev_string = '/device:GPU:0'
        else:
            dev_string = '/device:CPU:0'
        dev = tf.device(dev_string)
    print('Using device', dev_string)
    print('')

    print('---------------------Training Duplex-------------------')
    print('Author')
    print('Meta Materials Inc.')
    print('-------------------------------------------------------')

    with dev:
        if args.channel1:
            channels = 1
            mode = "grayscale"
        else:
            channels = 3
            mode = "rgb"

        model_dir = "model" + str(args.model_number)
        model = "model"
        model_path = os.path.join("..", "saved_models", model_dir,
                                  model + ".h5")
        if not os.path.isfile(model_path):
            raise ValueError("No Model" + str(args.model_number) +
                             " was created for " + str(channels) +
                             " channel(s) and " + str(args.image_size) +
                             " image size ")

        for i, arg in enumerate(vars(args)):
            print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
        print('-------------------------------------------------------\n')

        print("> Loading Model from ", model_path, " ...")
        custom_objects = {"PrunablePReLu": PrunablePReLu,
                          "PrunableConcat": PrunableConcat,
                          "PrunableDecompose1D": PrunableDecompose1D,
                          "PrunableClip": PrunableClip}

        model = load_model(model_path, custom_objects=custom_objects)

        root_dir = os.path.join("..", "data_" + args.order_pattern, "")
        rgb_directory = []
        mosaiced_directory = []
        for ifldr in args.dataset:
            nm_imgsz = ifldr + str(args.image_size)
            ds_nm = "dataset-" + ifldr
            tmpd = os.path.join(root_dir, "train", nm_imgsz, ds_nm, "")
            rgb_directory.append(tmpd)

            ds_nm += '-' + args.color_filter
            tmpd = os.path.join(root_dir, "train", nm_imgsz,
                                "data-" + str(channels) + "ch", ds_nm, "")
            mosaiced_directory.append(tmpd)

        print("> Loading RGB Dataset from ", *rgb_directory, " ...")
        print("> Loading Mosaiced Dataset from ", *mosaiced_directory, " ...")

        dir_to_save = os.path.join("..", "saved_models", model_dir, "")
        if distribute:
            out = training_session_distribute(
                strategy=strategy,
                model=model,
                directory=rgb_directory,
                mosaiced_directory=mosaiced_directory,
                color_mode=mode,
                image_size=args.image_size,
                scheduler=args.learning_rate_sch,
                dir_to_save=dir_to_save,
                batch_size=args.batch_size,
                epochs=args.epochs,
                floating_point=args.floating_point,
                train_metric=args.train_metric,
                continue_train=args.continue_train
            )
        else:
            out = training_session(model=model,
                                   directory=rgb_directory,
                                   mosaiced_directory=mosaiced_directory,
                                   color_mode=mode,
                                   image_size=args.image_size,
                                   scheduler=args.learning_rate_sch,
                                   dir_to_save=dir_to_save,
                                   batch_size=args.batch_size,
                                   epochs=args.epochs,
                                   floating_point=args.floating_point,
                                   train_metric=args.train_metric,
                                   continue_train=args.continue_train)
        trained_model, val_loss, val_ssim, val_cpsnr, train_loss, \
            train_ssim, train_cpsnr = out

    # if retrain, append to file
    dir_to_save = os.path.join(dir_to_save, "model")
    fflags = 'ab' if args.continue_train else 'wb'
    with open(dir_to_save + "_epochs_num.csv", fflags) as fh:
        np.savetxt(fh, np.array(args.epochs, ndmin=1), delimiter=",")
    with open(dir_to_save + "_val_loss.csv", fflags) as fh:
        np.savetxt(fh, val_loss, delimiter=",")
    with open(dir_to_save + "_val_ssim.csv", fflags) as fh:
        np.savetxt(fh, val_ssim, delimiter=",")
    with open(dir_to_save + "_val_cpsnr.csv", fflags) as fh:
        np.savetxt(fh, val_cpsnr, delimiter=",")

    with open(dir_to_save + "_train_loss.csv", fflags) as fh:
        np.savetxt(fh, train_loss, delimiter=",")
    with open(dir_to_save + "_train_ssim.csv", fflags) as fh:
        np.savetxt(fh, train_ssim, delimiter=",")
    with open(dir_to_save + "_train_cpsnr.csv", fflags) as fh:
        np.savetxt(fh, train_cpsnr, delimiter=",")

    trained_model.save(dir_to_save + "_trained.h5")

    dir_args = os.path.join("..", "saved_models", model_dir)
    with open(os.path.join(dir_args, 'args_train.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
