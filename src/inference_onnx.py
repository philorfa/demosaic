from argparse import ArgumentParser
import os
import json
import onnx
import onnxruntime as ort
import cfa_image as cfa_img
import cut_stitch_pics as csp
import numpy as np
from train_test_model import metrics_calc
from onnxconverter_common import float16
import time


def inference_data(sess, directory, cfa, channels,
                   image_size, save_dir, dataset, fp, order):

    ordering = order.upper()

    flist_in = cfa_img.pics_in_dir(directory)

    if fp == 16:
        cfa_img.fp_img = np.float16
    else:
        cfa_img.fp_img = np.float32

    input_name = sess.get_inputs()[0].name
    metrics_dataset = []
    results = []
    for image in flist_in:
        img_cut = csp.cut_frame(image, shape=image_size)
        dtype = img_cut[0][0].dtype

        img_fp_org = []
        img_fp_dms = []
        for row in img_cut:
            input_img = []
            for img in row:
                tmp_img = cfa_img.mosaic(img, cfa, order=ordering,
                                         one_channel=channels)
                input_img.append(cfa_img.convert2fp(tmp_img))
            input_img = np.stack(input_img, axis=0)
            if channels:
                input_img = np.expand_dims(input_img, axis=-1)
            demosaiced = sess.run(None, {input_name: input_img})[0]

            # keep original as float for metric calculations
            row = np.stack([cfa_img.convert2fp(img) for img in row], axis=0)
            img_fp_org.append(row)
            img_fp_dms.append(demosaiced)
            # img_fp_dms.append(np.clip(demosaiced, 0., 1.))

        image = os.path.split(image)[1]
        fp_org = np.vstack([np.stack(img, axis=0) for img in img_fp_org])
        fp_dms = np.vstack([np.stack(img, axis=0) for img in img_fp_dms])
        metrics = [float(mm) for mm in metrics_calc(fp_org, fp_dms)]
        case = {'image': image, 'l2': metrics[0],
                'ssim': metrics[1], 'cpsnr': metrics[2]}
        print(case)

        dms = [[cfa_img.convert2int(img, dtype) for img in rw]
               for rw in img_fp_dms]
        fname, fext = os.path.splitext(image)
        csp.stitch_list(dms, os.path.join(save_dir,
                                          fname + '_stitched' + fext))

        metrics_dataset.append(metrics)
        results.append(case)

    avg_total = np.average(metrics_dataset, axis=0)
    total = {'dataset': dataset, 'l2': avg_total[0],
             'ssim': avg_total[1], 'cpsnr': avg_total[2]}
    print(total)

    results_file = os.path.join(save_dir, "metrics.txt")
    with open(results_file, 'w+', encoding='utf-8') as file:
        for dic in results:
            data = json.dumps(dic)
            file.write(data)
            file.write("\n")
        file.write("\n\n\n")
        full = json.dumps(total)
        file.write(full)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('-mn', '--model_number', type=int, required=True)
    optional.add_argument('-pr', '--pruned', action='store_true')
    required.add_argument('-ds', '--dataset', type=str, required=True)
    optional.add_argument('-fp', '--onnx_fp', type=int, default=32,
                          help='Run FP32 or FP16 onnx model')
    optional.add_argument('-one', '--one_channel', action='store_true',
                          help="use the 3ch model that has been converted to 1ch")

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
    print('--------------------------------------------------------')

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

    model_dir = "model" + str(args.model_number)

    pruned = "_pruned" if args.pruned else ""

    # model_fp = "_fp32"
    # model_fp = "_uint8_tflite_dynamic"
    model_fp = "_quant"
    if args.onnx_fp == 16:
        model_fp = "_fp16"

    model = "model_1ch" + pruned + model_fp if args.one_channel else "model" + pruned + model_fp
    model_path = os.path.join("..", "saved_models_onnx",
                              model_dir, model + ".onnx")

    if not os.path.isfile(model_path):
        raise ValueError("No Model" + str(args.model_number) +
                         " was created for " + str(channels) +
                         " channel(s) and " + str(args.image_size) +
                         " image size")

    for i, arg in enumerate(vars(args)):
        print('{}.{}: {}'.format(i, arg, vars(args)[arg]))
    print('----------------------------------------------------------------\n')
    save_dir = os.path.join(root_dir + "\\inference_output\\" + args.dataset,
                            model_dir, model + "_onnx", "")

    print("> Loading Trained Model from ", model_path, " ...")
    prvdrs = ['CUDAExecutionProvider', ]
    ort_sess = ort.InferenceSession(model_path, providers=prvdrs)
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    rgb_directory = root_dir + "\\inference\\" + args.dataset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_time = time.time()
    if args.one_channel:
        args.channel1 = True
    inference_data(ort_sess, rgb_directory, args.color_filter, args.channel1,
                   args.image_size, save_dir, args.dataset,
                   args.onnx_fp, args.order_pattern)
    print("Execution time:", time.time() - start_time, "seconds.")
