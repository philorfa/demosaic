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
import onnx_tool
from onnx_tool import create_ndarray_f32
import errno
import shutil
import tempfile
import onnxruntime as ort
import time
from onnxconverter_common import float16
import onnx
import absl.logging

## https://www.tensorflow.org/lite/performance/post_training_quantization
## Parameters are quantized to float32. Model executes with float32 operations

absl.logging.set_verbosity(absl.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    parser = ArgumentParser(add_help=False)
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument('-mn', '--models', nargs="+", type=int,
                          help='Model(s) to profile and find throughput')
    optional.add_argument('-one', '--one_channel', action='store_true',
                         help = "test the 3ch model that has been converted to 1ch")

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

    custom_objects = {"PrunablePReLu": PrunablePReLu,
                      "PrunableConcat": PrunableConcat,
                      "PrunableDecompose1D": PrunableDecompose1D,
                      "PrunableClip": PrunableClip}
    try:

        
        options = ort.SessionOptions()
        options.enable_profiling = True
        
        tmp_dir = tempfile.mkdtemp()
        channels_list = []
        mac = []
        params = []

        milisc_fp32 = []
        mp_sec_fp32 = []
        
        milisc_fp16 = []
        mp_sec_fp16 = []

        batch_size = 16
        burn_in_iterations = 10
        num_iterations = 500

        for model_number in args.models:

       
            # Read model and files
            model = "model_1ch.h5" if args.one_channel else "profiler_model.h5"
            model_dir = os.path.join('..', 'saved_models', 'model' + str(model_number))
            model_path = os.path.join(model_dir, model)
            
            print("> Running Model ", model_number, " for FP32 and FP16 from",model_path,"...")
            
            with open(os.path.join(model_dir, 'args_create.txt'), 'r') as f:
                model_arguments = json.load(f)
            
            
            channels = 1 if model_arguments["channel1"] else 3
            channels_list.append(channels)

            model = load_model(model_path, custom_objects=custom_objects)

            # Convert to .pb and then to .onnx
            pb_path_32 = os.path.join(tmp_dir, "model" + str(model_number) + "_32")
            onnx_path_32 = os.path.join(tmp_dir, "model" + str(model_number) + "_32.onnx")

            if not os.path.exists(pb_path_32):
                os.makedirs(pb_path_32)

            # Convert to .pb
            model.save(pb_path_32)

            # Convert to .onnx
            argfile = "--saved-model"

            cmd = ["python", "-m", "tf2onnx.convert", argfile,
                   pb_path_32, "--output", onnx_path_32]
            subprocess.call(cmd)

            model_fp32 = onnx.load(onnx_path_32)
            
            my_op_block_list = ['ArrayFeatureExtractor', 'Binarizer', 'CastMap', 'CategoryMapper', 'DictVectorizer',
                         'FeatureVectorizer', 'Imputer', 'LabelEncoder', 'LinearClassifier', 'LinearRegressor',
                         'Normalizer', 'OneHotEncoder', 'RandomUniformLike', 'SVMClassifier', 'SVMRegressor', 'Scaler',
                         'TreeEnsembleClassifier', 'TreeEnsembleRegressor', 'ZipMap', 'NonMaxSuppression', 'TopK',
                         'RoiAlign', 'Resize', 'Range', 'CumSum', 'Upsample']
            
            model_fp16 = float16.convert_float_to_float16(model_fp32, op_block_list=my_op_block_list)
            
            onnx_path_16 = os.path.join(tmp_dir, "model" + str(model_number) + "_16.onnx")
            onnx.save(model_fp16, onnx_path_16)
            
            # Starting profiling session 
            
            prvdrs = ['CUDAExecutionProvider', ]
            session = ort.InferenceSession(onnx_path_32, providers=prvdrs)
            input_name = session.get_inputs()[0].name
            
            if args.one_channel:
                channels = 1
            
            inputshape = (1, model_arguments["image_size"], model_arguments["image_size"], channels)
            dynamics_input = {input_name: create_ndarray_f32(inputshape)}
            output_file = os.path.join('..', 'saved_models', 'model' + str(model_number), 'profiler.txt')

            with open(output_file, 'w+') as f:
                with redirect_stdout(f):
                    onnx_tool.model_profile(onnx_path_32, dynamic_shapes=dynamics_input)

            with open(output_file) as f:
                for line in f:
                    if line.startswith("Total"):
                        op = line.split()[2]
                        mac.append(op)
                        break

            params.append(model_arguments["total_params"])
            if args.one_channel:
                model_mac = {'MACs_1ch': op}
            else:
                model_mac = {'MACs': op}
            model_arguments.update(model_mac)

            with open(os.path.join(model_dir, 'args_create.txt'), 'w+') as f:
                json.dump(model_arguments, f, indent=2)

            # Starting execution time for fp32   
            input_shape = (batch_size, model_arguments["image_size"], model_arguments["image_size"], channels)
            input_data = np.random.rand(*input_shape).astype(np.float32)
            
            # Warm up the model with burn-in iterations
            for i in range(burn_in_iterations):
                session.run(None, {input_name: input_data})

            # Benchmark the model
            total_time = 0
            for i in range(num_iterations):
                start_time = time.time()
                session.run(None, {input_name: input_data})
                end_time = time.time()
                total_time += end_time - start_time
            
            average_time = total_time / num_iterations
            milisc_fp32.append(total_time / num_iterations)
            pixels_per_second = batch_size * 128 * 128 / average_time
            mp_sec_fp32.append(pixels_per_second * 10 ** -6)
            
            
            # Starting execution time for fp16
            
            prvdrs = ['CUDAExecutionProvider', ]
            session = ort.InferenceSession(onnx_path_16,options, providers=prvdrs)
            input_name = session.get_inputs()[0].name
            
            input_shape = (batch_size, model_arguments["image_size"], model_arguments["image_size"], channels)
            input_data = np.random.rand(*input_shape).astype(np.float16)
            
            # Warm up the model with burn-in iterations
            for i in range(burn_in_iterations):
                session.run(None, {input_name: input_data})

            # Benchmark the model
            total_time = 0
            for i in range(num_iterations):
                start_time = time.time()
                session.run(None, {input_name: input_data})
                end_time = time.time()
                total_time += end_time - start_time
            
            average_time = total_time / num_iterations
            milisc_fp16.append(total_time / num_iterations)
            pixels_per_second = batch_size * 128 * 128 / average_time
            mp_sec_fp16.append(pixels_per_second * 10 ** -6)
            
            prof_file = session.end_profiling()
            print("fp16 file:", prof_file)

        table_data = [['Model', 'Parameters', 'MACs', 'Exec Time (ms) FP32', 'MP/sec FP32', 
                       'Exec Time (ms) FP16', 'MP/sec FP16']]
        for i, model in enumerate(args.models):
            data = [str(model), params[i], mac[i], milisc_fp32[i], mp_sec_fp32[i], milisc_fp16[i], mp_sec_fp16[i]]
            table_data = np.vstack([table_data, data])

        for row in table_data:
            print("{: <5} {: <12} {: <15} {: <25} {: <20} {: <24} {: <15}".format(*row))

    finally:
        try:
            shutil.rmtree(tmp_dir)  # delete directory
        except OSError as exc:
            if exc.errno != errno.ENOENT:  # ENOENT - no such file or directory
                raise  # re-raise exception


