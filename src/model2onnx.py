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
from onnxconverter_common import float16
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import absl.logging
from onnxruntime.quantization.shape_inference import quant_pre_process

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
absl.logging.set_verbosity(absl.logging.ERROR)

if __name__ == '__main__':

    parser = ArgumentParser(add_help=False)
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('-mn', '--model_number', required=True, type=int)
    optional.add_argument('-pr', '--pruned', action='store_true')
    optional.add_argument('-eptr', '--epoch_training', default = 0, type=int)
    optional.add_argument('-one', '--one_channel', action='store_true',
                         help = "use the 3ch model that has been converted to 1ch")
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
    model = "model_1ch" if args.one_channel else "model_trained"
    pruned = ""  
    if args.pruned:
        model_path = os.path.join("..", "saved_models_pruned",
                                  model_dir, model + ".h5")
        pruned = "_pruned"
    else:
        model_path = os.path.join("..", "saved_models", model_dir,
                                  model + ".h5")
    if args.epoch_training:
        model_path = os.path.join("..", "saved_models",
                                  model_dir, "model_" + str(args.epoch_training) + "epochs.h5")
    

    print("> Loading Model from ", model_path, " ...")
    custom_objects = {"PrunablePReLu": PrunablePReLu, 
                      "PrunableConcat": PrunableConcat, 
                      "PrunableDecompose1D": PrunableDecompose1D, 
                      "PrunableClip": PrunableClip}
    model = load_model(model_path, custom_objects=custom_objects)

    pb_path = os.path.join("..", "saved_models_onnx", model_dir)
    if not os.path.exists(pb_path):
        os.makedirs(pb_path)
        
    
    print("> Converting to FP32 onnx ...")
    
    model.save(pb_path)
    
    model = "model_1ch" if args.one_channel else "model"
    onnx_path_fp32 = os.path.join(pb_path, model + pruned +"_fp32.onnx")
        
    argfile = "--saved-model"

    cmd = ["python", "-m", "tf2onnx.convert", argfile,
           pb_path, "--output", onnx_path_fp32]
    subprocess.call(cmd)
    
    print("> Converting to FP16 onnx ...")
    
    model_fp32 = onnx.load(onnx_path_fp32)
    
    my_op_block_list = ['ArrayFeatureExtractor', 'Binarizer', 'CastMap', 'CategoryMapper', 'DictVectorizer',
                         'FeatureVectorizer', 'Imputer', 'LabelEncoder', 'LinearClassifier', 'LinearRegressor',
                         'Normalizer', 'OneHotEncoder', 'RandomUniformLike', 'SVMClassifier', 'SVMRegressor', 'Scaler',
                         'TreeEnsembleClassifier', 'TreeEnsembleRegressor', 'ZipMap', 'NonMaxSuppression', 'TopK',
                         'RoiAlign', 'Resize', 'Range', 'CumSum', 'Upsample']
    model_fp16 = float16.convert_float_to_float16(model_fp32, op_block_list=my_op_block_list)
    
    model = "model_1ch" if args.one_channel else "model"
    
    onnx_path_fp16 = os.path.join(pb_path, model + pruned + "_fp16.onnx")
    onnx.save(model_fp16, onnx_path_fp16)
    
    onnx_path_pre = os.path.join(pb_path, model + pruned + "_pre.onnx")
    quant_pre_process(onnx_path_fp32,onnx_path_pre)
    
    onnx_path_int8 = os.path.join(pb_path, model + pruned + "_quant.onnx")
    quantized_model = quantize_dynamic(onnx_path_pre, onnx_path_int8, weight_type=QuantType.QUInt8)
    
    print("> Done")
