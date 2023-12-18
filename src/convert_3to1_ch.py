import sys
from argparse import ArgumentParser
import os
from os import path
from model import *
import json
from keras.models import load_model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    parser = ArgumentParser(add_help=False)
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('-mn', '--model_number', required=True, type=int,
                          help='identity number of your model')
    
    
    
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

    print('---------------------Welcome to Duplex-------------------')
    print('Author')
    print('Meta Materials Inc.')
    print('---------------------------------------------------------------------')

    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('---------------------------------------------------------------------\n')

    model = "model_trained"
    model_path = os.path.join(model_dir, model + ".h5")
    if not os.path.isfile(model_path):
        raise ValueError("No Model" + str(args.model_number) +
                         " was created for " + str(channels) +
                         " channel(s) and " + str(args.image_size) +
                         " image size ")

    print("> Loading Model from ", model_path, " ...")
    custom_objects = {"PrunablePReLu": PrunablePReLu, 
                      "PrunableConcat": PrunableConcat, 
                      "PrunableDecompose1D": PrunableDecompose1D, 
                      "PrunableClip": PrunableClip}
    
    model = load_model(model_path, custom_objects=custom_objects)
    size = (128, 128, 1)
    input = Input(size)
    x = PrunableDecompose1D("chame")(input)
    output = model(x)
    model = keras.models.Model(input, output)
    
    print('---------------------------------------------------------------------')
    x = tf.ones((1, 128, 128, 1), dtype="float32")
    print("Input dtype:", x.dtype)
    print('---------------------------------------------------------------------')
    print("Output Inference dtype: ",model(x).dtype)
    print("output Training dtype: ",model(x, training = True).dtype)
    print('---------------------------------------------------------------------')
    for layer in model.layers[-5:]:
        print("Name of layer: ", layer.name)
        print("Layer's dtype: ", layer.dtype)
        print("Layer's Compute dtype: ", layer.compute_dtype)
        #print("Layer's Weights: ", layer.weights)
        print('---------------------------------------------------------------------')
        
    model.summary()
    model.save(os.path.join(model_dir, 'model_1ch.h5'))