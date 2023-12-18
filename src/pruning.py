import sys
from argparse import ArgumentParser
from model import *
from keras.models import load_model
from train_test_model import *
from util import *
from os import path

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.config.list_physical_devices('GPU'))

if __name__ == '__main__':

    parser = ArgumentParser(add_help=False)
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    required.add_argument('-mn', '--model_number', type=int, required=True)
    optional.add_argument('-ds', '--dataset', default='medium', type=str)
    optional.add_argument('-ssp', '--start_sparsity', default=50, type=int)
    optional.add_argument('-esp', '--end_sparsity', default=80, type=int)
    optional.add_argument('-e', '--epochs', default=5, type=int)
    optional.add_argument('-bs', '--batch_size', default=16, type=int)
    optional.add_argument('-eptr', '--epoch_training', default=0, type=int,
                          help='in case, training was interrupted, and we want to test saved models')
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

    try:
        pass
    except:
        sys.exit(0)

    print('---------------------Training Duplex-------------------')
    print('Author')
    print('Meta Materials Inc.')
    print('---------------------------------------------------------------------')

    
    if args.color_filter not in ['bayer', 'nona', 'tetra', 'chame']:
        raise ValueError("CFA is not compatible")
   

    if args.channel1:
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
        model = "model_trained"
        model_epoch = ""
    model_path = os.path.join("..", "saved_models", model_dir, model + model_epoch + ".h5")
    
    if not os.path.isfile(model_path):
        raise ValueError("No Model" + str(args.model_number) +
                         " was created for " + str(channels) +
                         " channel(s) and " + str(args.image_size) +
                         " image size")

    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('---------------------------------------------------------------------\n')

    
    dir_to_save = os.path.join("..", "saved_models_pruned", model_dir, model + model_epoch)
    if not path.isdir(os.path.join("..", "saved_models_pruned", model_dir)):
        os.mkdir(os.path.join("..", "saved_models_pruned", model_dir))
        
    print("> Loading Trained Model from ", model_path," ...")
    
    custom_objects = {"PrunablePReLu": PrunablePReLu, 
                      "PrunableConcat": PrunableConcat, 
                      "PrunableDecompose1D": PrunableDecompose1D, 
                      "PrunableClip": PrunableClip}
    
    model = load_model(model_path, custom_objects=custom_objects)

    rgb_directory = "..\\data\\train\\" + args.dataset + str(args.image_size) + "\\dataset-" + args.dataset
    mosaiced_directory = "..\\data\\train\\" + args.dataset + str(
        args.image_size) + "\\data-" + str(channels) + "ch\\dataset-" + args.dataset + "-" + args.color_filter

    print("> Loading Dataset from ", rgb_directory," ...")
    print("> Loading Mosaiced Dataset from ", mosaiced_directory," ...")
    
    pruned_model, val_loss, val_ssim, val_cpsnr, train_loss, train_ssim, train_cpsnr = prune(model,
                                                                                              rgb_directory,
                                                                                              mosaiced_directory,
                                                                                              mode,
                                                                                              args.start_sparsity,
                                                                                              args.end_sparsity,
                                                                                              args.batch_size,
                                                                                              args.epochs,
                                                                                              args.floating_point)
    

    np.savetxt(dir_to_save + "_val_loss.csv", val_loss, delimiter=",")
    np.savetxt(dir_to_save + "_val_ssim.csv", val_ssim, delimiter=",")
    np.savetxt(dir_to_save + "_val_cpsnr.csv", val_cpsnr, delimiter=",")

    np.savetxt(dir_to_save + "_train_loss.csv", train_loss, delimiter=",")
    np.savetxt(dir_to_save + "_train_ssim.csv", train_ssim, delimiter=",")
    np.savetxt(dir_to_save + "_train_cpsnr.csv", train_cpsnr, delimiter=",")

    model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)
    
    tf.keras.models.save_model(model_for_export, dir_to_save + ".h5" , include_optimizer=False)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    pruned_tflite_model = converter.convert()
    
    with open(dir_to_save + ".tflite", 'wb') as f:
        f.write(pruned_tflite_model)
            
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_and_pruned_tflite_model = converter.convert()
    
    
    with open(dir_to_save + "quant.tflite", 'wb') as f:
        f.write(quantized_and_pruned_tflite_model)
        
    dir_args = os.path.join("..", "saved_models_pruned", model_dir)
    with open(os.path.join(dir_args, 'args_pruned.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        