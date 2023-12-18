from argparse import ArgumentParser
import os
import tensorflow as tf
import json
import time
from keras.models import load_model
from train_test_model import inference_data
from model import PrunablePReLu, PrunableConcat
from model import PrunableDecompose1D, PrunableClip


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.config.list_physical_devices('GPU'))


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('-mn', '--model_number', type=int, required=True)
    optional.add_argument('-pr', '--pruned', action='store_true')
    optional.add_argument('-eptr', '--epoch_training', default=0, type=int,
                          help='in case, training was interrupted, and we want to inference saved models')
    required.add_argument('-ds', '--dataset', type=str, required=True)
    optional.add_argument('-one', '--one_channel', action='store_true',
                          help="use the 3ch model that has been converted to 1ch")
    optional.add_argument('-fp16', '--convert-float16', action='store_true',
                          help='Create TFLite FP16 model and run inference')
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        help='show this help message and exit'
    )

    args = parser.parse_args()
    model_dir = os.path.join('..', 'saved_models',
                             'model' + str(args.model_number))

    dataset = args.dataset
    with open(os.path.join(model_dir, 'args_train.txt'), 'r') as f:
        args.__dict__.update(json.load(f))
    args.dataset = dataset
    print('---------------------Inference Duplex-------------------')
    print('Author')
    print('Meta Materials Inc.')
    print('---------------------------------------------------------------------')

    if args.color_filter not in ['bayer', 'nona', 'tetra', 'chame']:
        raise ValueError("CFA is not compatible")

    root_dir = "..\\data_" + args.order_pattern

    # load and check the datasets
    root = root_dir + "\\inference"
    datasets = []
    for item in os.listdir(root):
        if os.path.isdir(os.path.join(root, item)):
            datasets.append(item)

    if args.dataset not in datasets:
        raise ValueError("Unknown Dataset", str(args.dataset))

    channels = 1 if args.channel1 else 3

    tofp16 = args.convert_float16
    fp16 = "_fp16lite" if tofp16 else ""

    model_dir = "model" + str(args.model_number)

    if args.epoch_training:
        model = "model"
        model_epoch = "_" + str(args.epoch_training) + "epochs"
    else:
        model = "model_1ch" if args.one_channel else "model_trained"
        model_epoch = ""

    if args.pruned:
        model_path = os.path.join("..", "saved_models_pruned",
                                  model_dir, model + model_epoch + ".h5")
    else:
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

    if args.pruned:
        save_dir = os.path.join(root_dir + "\\inference_output\\" + args.dataset,
                                model_dir + "_pruned" + fp16, model + model_epoch)
    else:
        save_dir = os.path.join(root_dir + "\\inference_output\\" + args.dataset,
                                model_dir + fp16, model + model_epoch)

    print("> Loading Trained Model from ", model_path, " ...")

    custom_objects = {"PrunablePReLu": PrunablePReLu,
                      "PrunableConcat": PrunableConcat,
                      "PrunableDecompose1D": PrunableDecompose1D,
                      "PrunableClip": PrunableClip}

    model = load_model(model_path, custom_objects=custom_objects)

    if tofp16:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        fname, fext = os.path.splitext(model_path)
        model = fname + '.tflite'
        with open(model, 'bw') as ftfl:
            ftfl.write(tflite_model)

    rgb_directory = root_dir + "\\inference\\" + args.dataset

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_time = time.time()
    if args.one_channel:
        args.channel1 = True
    inference_data(model, rgb_directory, args.color_filter, args.channel1,
                   args.image_size, save_dir, args.dataset,
                   args.floating_point, args.order_pattern, tofp16)
    print("Execution time:", time.time() - start_time, "seconds.")
