import sys
from argparse import ArgumentParser
import os
from os import path
from model import *
import json
import shutil

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    parser = ArgumentParser(add_help=False)
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('-mn', '--model_number', required=True, type=int,
                          help='identity number of your model')
    required.add_argument('-pl', '--pyramid_level', default=4, type=int)
    optional.add_argument('-ch', '--channel1', action='store_true',
                          help='1 or 3 channels for the input image')
    optional.add_argument('-actfun', '--activation_function', default='prelu',
                          type=str, help='relu or prelu')
    optional.add_argument('-sz', '--image_size', default=128, type=int,
                          help='image resolution e.g. 128 -> (128,128)')
    optional.add_argument('-cf', '--color_filter', default="chame", type=str)
    optional.add_argument('-stride', '--strided_model', action='store_true',
                          help='larger stride in first and last layer')
    optional.add_argument('-a', '--alpha', default=16, type=int)
    optional.add_argument('-b', '--beta', default=32, type=int)
    optional.add_argument('-pol', '--model_policy', default="float32", type=str, 
                          help='float16, float32, mixed_float16')
    optional.add_argument('-fp', '--floating_point', default="float32",
                          type=str, help='Data dtype float16 or float32')
    optional.add_argument('-dc', '--decompose', action='store_true',
                          help='use decompose layer')
    
    
    
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        help='show this help message and exit'
    )

    try:
        args = parser.parse_args()
    except Exception:
        sys.exit(0)

    print('---------------------Welcome to Duplex-------------------')
    print('Author')
    print('Meta Materials Inc.')
    print('---------------------------------------------------------')

    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('---------------------------------------------------------\n')

    if args.color_filter not in ['bayer', 'nona', 'tetra', 'chame']:
        raise ValueError("CFA is not compatible")
    if args.activation_function not in ['relu', 'prelu']:
        raise ValueError("Activation function is not compatible")

    print("> Creating Duplex Model ... ")
    function = 0
    if args.activation_function == 'relu':
        function = 1
    if args.channel1:
        channels = 1
        model = duplex_1ch(input_size=args.image_size,
                               cfa=args.color_filter,
                               pyramid_level=args.pyramid_level,
                               function=function,
                               alpha = args.alpha,
                               beta = args.beta,
                               model_policy = args.model_policy,
                               stride = args.strided_model,
                               decompose = args.decompose,
                               bias = 0)
        profiler_model = duplex_1ch(input_size=args.image_size,
                               cfa=args.color_filter,
                               pyramid_level=args.pyramid_level,
                               function=function,
                               alpha = args.alpha,
                               beta = args.beta,
                               model_policy = args.model_policy,
                               stride = args.strided_model,
                               decompose = args.decompose,
                               bias = 1)
            
    else:
        channels = 3
        model = duplex_3ch(input_size=args.image_size, 
                           pyramid_level = args.pyramid_level, 
                           function=function,
                           alpha = args.alpha, 
                           beta = args.beta, 
                           model_policy = args.model_policy, 
                           stride = args.strided_model,
                           bias = 0)
        profiler_model = duplex_3ch(input_size=args.image_size, 
                           pyramid_level = args.pyramid_level, 
                           function=function,
                           alpha = args.alpha, 
                           beta = args.beta, 
                           model_policy = args.model_policy, 
                           stride = args.strided_model,
                           bias = 1)
 
    
    print('---------------------------------------------------------------------')
    x = tf.ones((channels, 128, 128, channels), dtype=args.floating_point)
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
   
    marks = {'total_params':model.count_params()}
    args.__dict__.update(marks)
    print("> Saving model ...")
    model_dir = os.path.join('..', 'saved_models',
                             'model' + str(args.model_number))
    if path.isdir(model_dir):
        shutil.rmtree(model_dir)
        
    os.mkdir(model_dir)
        
    model.save(os.path.join(model_dir, 'model.h5'))
    profiler_model.save(os.path.join(model_dir, 'profiler_model.h5'))
    with open(os.path.join(model_dir, 'args_create.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

