from argparse import ArgumentParser
from keras.models import load_model
from model import PrunablePReLu, PrunableConcat, PrunableDecompose1D, PrunableClip
import os
import subprocess
import sys
import tensorflow as tf
import json
from contextlib import redirect_stdout
from image_dataset import *

## https://www.tensorflow.org/lite/performance/post_training_quantization
## Parameters are quantized to float32. Model executes with float32 operations


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    parser = ArgumentParser(add_help=False)
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('-mn', '--model_number', required=True, type=int)
    optional.add_argument('-pr', '--pruned', action='store_true')
    optional.add_argument('-eptr', '--epoch_training', default=0, type=int,
                          help='in case of continue training, select from the saved models, the epoch to continue')
    optional.add_argument('-conv', '--convert', default=None, type=str,
                          help='Convert to TFLite FP16 or uint8 model before export')
    optional.add_argument('-qt', '--quant_type', default='static', type=str,
                          help='static or dynamic quantization if uint8 is selected as convert argument')
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

    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('----------------------------------------------------------------\n')
    
    
    model_dir = "model" + str(args.model_number)
    if args.epoch_training:
        model = "model_"
        model_epoch = str(args.epoch_training)
    else:
        model = "model_trained"
        model_epoch = ""
    pruned = ""  
    if args.pruned:
        model_path = os.path.join("..", "saved_models_pruned",
                                  model_dir, model + model_epoch + ".h5")
        pruned = "_pruned"
    else:
        model_path = os.path.join("..", "saved_models", model_dir,
                                  model + model_epoch + ".h5")
    

    print("> Loading Model from ", model_path, " ...")
    custom_objects = {"PrunablePReLu": PrunablePReLu, 
                      "PrunableConcat": PrunableConcat, 
                      "PrunableDecompose1D": PrunableDecompose1D, 
                      "PrunableClip": PrunableClip}
    model = load_model(model_path, custom_objects=custom_objects)
    
    
    if args.convert == "fp16":
        convert = "_fp16"
    if args.convert == "uint8":
        convert = "_uint8"
    if args.convert == None:
        convert = "_fp32"

    model_path_med_output = os.path.join("..", "saved_models_onnx", model_dir)
    if not os.path.exists(model_path_med_output):
        os.makedirs(model_path_med_output)

    if args.convert == "uint8":
        
        print("> Converting to TFLite uint8 format ...")
        model_path_input = os.path.join(model_path_med_output, "model" + pruned + convert + ".tflite")
        
        if args.quant_type == 'static':    
        # Create representative_data_gen       
        
            dataset = "extra"
            channels = 3
            mode = "rgb"
            if args.channel1:
                channels = 1
                mode = "grayscale"

            mosaiced_directory = "..\\data\\test\\" + \
                                 dataset + \
                                 str(args.image_size) + \
                                 "\\data-" + \
                                 str(channels) + \
                                 "ch\\dataset-" + \
                                 dataset + \
                                 "-" + \
                                 args.color_filter


            with redirect_stdout(None):

                calibration_ds = image_dataset_from_directory(
                                    mosaiced_directory,
                                    args.floating_point,
                                    label_mode=None,
                                    color_mode=mode,
                                    seed=123,
                                    batch_size=1,
                                    shuffle=False)


            def representative_data_gen():
                  for input_value in calibration_ds.take(100):
                        yield [input_value]

            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_data_gen

            # Ensure that if any ops can't be quantized, the converter throws an error
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

            # Set the input and output tensors to int8 
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8

            tflite_model = converter.convert()

            with open(model_path_input, 'bw') as ftfl:
                ftfl.write(tflite_model)

            argfile = "--tflite"

            model_path_output = os.path.join(model_path_med_output, "model" + pruned + convert +"_tflite_static.onnx")
        else:
            
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
        
            
            with open(model_path_input, 'bw') as ftfl:
                ftfl.write(tflite_model)

            argfile = "--tflite"

            model_path_output = os.path.join(model_path_med_output, "model" + pruned + convert +"_tflite_dynamic.onnx")
        
        print("> Converting .tflite to onnx format")
        
    elif args.convert == "fp16":
        
        print("> Converting to TFLite FP16 format ...")
        model_path_input = os.path.join(model_path_med_output, "model" + pruned + convert + ".tflite")
       
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        with open(model_path_input, 'bw') as ftfl:
            ftfl.write(tflite_model)

        argfile = "--tflite"
        
        model_path_output = os.path.join(model_path_med_output, "model" + pruned + convert +"_tflite.onnx")
        print("> Converting .tflite to onnx format")
    
    elif args.convert == None:
        
        print("> Saving model to .pb format ...")
        model.save(model_path_med_output)
        
        model_path_input = model_path_med_output
        
        model_path_output = os.path.join(model_path_input, "model" + pruned + convert +".onnx")
        
        
        argfile = "--saved-model"
        print("> Convert .pb to onnx format")

    print("> Saving file to ", model_path_output," ...")
    cmd = ["python", "-m", "tf2onnx.convert", argfile,
           model_path_input, "--output", model_path_output]
    subprocess.call(cmd)
    print("> Done")
