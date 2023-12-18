from argparse import ArgumentParser
from keras.models import load_model
from train_test_model import testing_session
from model import PrunablePReLu, PrunableConcat
from model import PrunableDecompose1D, PrunableClip
import tensorflow as tf
import os
import json

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.config.list_physical_devices('GPU'))

if __name__ == '__main__':

    parser = ArgumentParser(add_help=False)
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('-mn', '--model_number', type=int, required=True)
    optional.add_argument('-ds', '--dataset', default='extra', type=str)
    optional.add_argument('-bs', '--batch_size', default=16, type=int)
    optional.add_argument('-one', '--one_channel', action='store_true',
                          help='test the 3ch model that has been converted'
                               ' to 1ch')
    optional.add_argument('-eptr', '--epoch_training', default=0, type=int,
                          help='in case, training was interrupted, and we want'
                               ' to test saved models')
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        help='show this help message and exit'
    )

    args = parser.parse_args()

    model_dir = os.path.join('..', 'saved_models',
                             'model' + str(args.model_number))

    with open(os.path.join(model_dir, 'args_train.txt'), 'r') as f:
        args.__dict__.update(json.load(f))

    print('---------------------Testing Duplex---------------------------')
    print('Author')
    print('Meta Materials Inc.')
    print('--------------------------------------------------------------')

    if args.color_filter not in ['bayer', 'nona', 'tetra', 'chame']:
        raise ValueError("CFA is not compatible")

    if args.channel1 or args.one_channel:
        channels = 1
        mode = "grayscale"
    else:
        channels = 3
        mode = "rgb"

    model_dir = "model" + str(args.model_number)
    if args.epoch_training:
        model = "model"
        model_epoch = "_" + str(args.epoch_training) + "epochs"
    else:
        model = "model_1ch" if args.one_channel else "model_trained"
        model_epoch = ""
    model_path = os.path.join("..", "saved_models", model_dir,
                              model + model_epoch + ".h5")

    if not os.path.isfile(model_path):
        raise ValueError("No Model" + str(args.model_number) +
                         " was created for " + str(channels) +
                         " channel(s) and " + str(args.image_size) +
                         " image size")

    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('----------------------------------------------------------------\n')

    print("> Loading Trained Model from ", model_path, " ...")

    custom_objects = {"PrunablePReLu": PrunablePReLu,
                      "PrunableConcat": PrunableConcat,
                      "PrunableDecompose1D": PrunableDecompose1D,
                      "PrunableClip": PrunableClip}

    model = load_model(model_path, custom_objects=custom_objects)

    root_dir = "..\\data_" + args.order_pattern

    rgb_directory = root_dir + "\\test\\" + args.dataset \
        + str(args.image_size) + "\\dataset-" + args.dataset

    mosaiced_directory = root_dir + "\\test\\" + args.dataset \
        + str(args.image_size) + "\\data-" + str(channels) + "ch\\dataset-" \
        + args.dataset + "-" + args.color_filter

    print("> Loading Dataset from ", rgb_directory, " ...")
    print("> Loading Mosaiced Dataset from ", mosaiced_directory, " ...")

    testing_session(model, rgb_directory, mosaiced_directory, mode,
                    args.batch_size, args.floating_point)
