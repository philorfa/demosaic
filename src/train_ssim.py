from tensorflow_addons.optimizers import AdamW
import collections
import math
from contextlib import redirect_stdout
import cut_stitch_pics as csp
import cfa_image as cfa_img
import tensorflow as tf
import numpy as np
from image_dataset import image_dataset_from_directory
import os
import json

import tensorflow_model_optimization as tfmot
from pathlib import Path
from os import path
import time

AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def ssim_loss(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.))

@tf.function
def custom_mse_loss(y_true, y_pred):
    square_error = tf.math.square(y_true - y_pred)
    mean_square_error = tf.reduce_mean(square_error)
    return mean_square_error

@tf.function
def cpsnr(original, contrast):
    mse = custom_mse_loss(original, contrast)
    if mse == 0:
        return tf.convert_to_tensor(100., dtype=mse.dtype)
    return -10. * tf.experimental.numpy.log10(mse)


@tf.function
def custom_l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.abs(y_true - y_pred))

@tf.function
def metrics_calc(y_true, y_pred):
    
    l2_value = custom_mse_loss(y_true, y_pred)
    #l1_value = custom_l1_loss(y_true, y_pred)
    ssim_value = ssim_loss(y_true, y_pred)
    cpsnr_value = cpsnr(y_true, y_pred)
    
    
    return [l2_value, ssim_value, cpsnr_value]

@tf.function
def train_step(model, x_batch_train, y_batch_train, optimizer):
    with tf.GradientTape() as tape:
        logits = model(x_batch_train, training=True)
        metrics_train = metrics_calc(y_batch_train, logits)
        #loss_train = -0.84 * metrics_train[1] + (1 - 0.84) * metrics_train[0]
        loss_train = metrics_train[0]

    grads = tape.gradient(loss_train, model.trainable_weights)

    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return metrics_train, loss_train


def training_session(model, directory, mosaiced_directory, color_mode,
                     scheduler, dir_to_save, batch_size=16, epochs=5,
                     epoch_training=0, floating_point="float32", patience=3,
                     train_metric="l2", learning_rate=0.0001):
    # READ DATA
    with redirect_stdout(None):
        target_train_ds = image_dataset_from_directory(
            directory,
            floating_point,
            label_mode=None,
            validation_split=0.2,
            subset="training",
            seed=123,
            batch_size=batch_size,
            shuffle=False)

        input_train_ds = image_dataset_from_directory(
            mosaiced_directory,
            floating_point,
            label_mode=None,
            color_mode=color_mode,
            validation_split=0.2,
            subset="training",
            seed=123,
            batch_size=batch_size,
            shuffle=False)

        target_val_ds = image_dataset_from_directory(
            directory,
            floating_point,
            label_mode=None,
            validation_split=0.2,
            subset="validation",
            seed=123,
            batch_size=batch_size,
            shuffle=False)

        input_val_ds = image_dataset_from_directory(
            mosaiced_directory,
            floating_point,
            color_mode=color_mode,
            label_mode=None,
            validation_split=0.2,
            subset="validation",
            seed=123,
            batch_size=batch_size,
            shuffle=False)
    
    # CREATE FILE FOR LIVE TRACKING
    
    out_path = Path(dir_to_save)
    out_file = os.path.join(out_path.parent.absolute(), 'train_live.txt')
    if not path.isfile(out_file):
        open(out_file, 'a').close()

    # PREPARE DATA
    target_train_ds = target_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    target_validation_ds = target_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    input_train_ds = input_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    input_validation_ds = input_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    validation_ds = tf.data.Dataset.zip(
        (target_validation_ds, input_validation_ds))
    train_ds = tf.data.Dataset.zip((target_train_ds, input_train_ds))

    # PREPARE VALUES FOR PROGBAR
    validation_samples = len(target_val_ds)
    training_samples = len(target_train_ds)

    num_samples = batch_size * (validation_samples + training_samples)

    metrics_names = ['train_loss', 'val_loss', 'CPSNR', 'SSIM', 'L1']

    # PREPARE LISTS FOR MONITORING LOSSES
    loss_list_per_batch = collections.deque()
    ssim_list_per_batch = collections.deque()
    cpsnr_list_per_batch = collections.deque()

    loss_list_per_batch_val = collections.deque()
    ssim_list_per_batch_val = collections.deque()
    cpsnr_list_per_batch_val = collections.deque()

    train_loss = []
    val_loss = []

    train_ssim = []
    val_ssim = []

    train_cpsnr = []
    val_cpsnr = []

    # PREPARE VALUES FOR OPTIMIZER AND LEARNING RATE SCHEDULER
    optimizer = AdamW(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999,
                      weight_decay=1e-4)
    if epoch_training != 0:
        
        o_weights = np.load((dir_to_save + "_" + str(epoch_training)
                                     + "epochs" + '_optimizer.npz'),
                                    allow_pickle=True)
        optimizer_weights = [o_weights[k] for k in o_weights]
        
        model_train_vars = model.trainable_weights
        zero_grads = [tf.zeros_like(w) for w in model_train_vars]

        # save current state of variables
        saved_vars = [tf.identity(w) for w in model_train_vars]

        # Apply gradients which don't do nothing
        optimizer.apply_gradients(zip(zero_grads, model_train_vars))

        # Reload variables
        [x.assign(y) for x, y in zip(model_train_vars, saved_vars)]

        # Set the weights of the optimizer
        optimizer.set_weights(optimizer_weights)
    else:
        
        model_train_vars = model.trainable_weights
        zero_grads = [tf.zeros_like(w) for w in model_train_vars]

        # save current state of variables
        saved_vars = [tf.identity(w) for w in model_train_vars]

        # Apply gradients which don't do nothing
        optimizer.apply_gradients(zip(zero_grads, model_train_vars))

        # Reload variables
        [x.assign(y) for x, y in zip(model_train_vars, saved_vars)]
        
    
    
    loss_patience = np.Inf
    patience_loop = patience

    for epoch in range(epochs):
        
        epoch_time_start = time.time()
        
        str1 = "\nStart of epoch " + str(int(epoch + 1 + epoch_training))
        print(str1)
        progBar = tf.keras.utils.Progbar(num_samples,
                                         stateful_metrics=metrics_names)

        # Training loop
        for it, (y_batch_train, x_batch_train) in enumerate(train_ds):
            
            metrics_train, loss_train = train_step(model, x_batch_train,
                                              y_batch_train, optimizer)

            loss_list_per_batch.append(loss_train)
            ssim_list_per_batch.append(metrics_train[1])
            cpsnr_list_per_batch.append(metrics_train[2])

            values = [('train_loss', loss_train),
                      ('CPSNR', metrics_train[2]), 
                      ('SSIM', metrics_train[1])]
            progBar.update((it + 1) * batch_size, values=values)

        epoch_loss_train = np.average(loss_list_per_batch)
        train_loss.append(epoch_loss_train)

        epoch_ssim_train = np.average(ssim_list_per_batch)
        train_ssim.append(epoch_ssim_train)

        epoch_cpsnr_train = np.average(cpsnr_list_per_batch)
        train_cpsnr.append(epoch_cpsnr_train)

        loss_list_per_batch.clear()
        ssim_list_per_batch.clear()
        cpsnr_list_per_batch.clear()

        # Validation loop
        for iv, (y_batch_val, x_batch_val) in enumerate(validation_ds):
            logits_val = model(x_batch_val)
            metrics_val = metrics_calc(y_batch_val, logits_val)
            
            #loss_val = -0.84 * metrics_val[1] + (1 - 0.84) * metrics_val[0]
            loss_val = metrics_val[0]
            
            loss_list_per_batch_val.append(loss_val)
            ssim_list_per_batch_val.append(metrics_val[1])
            cpsnr_list_per_batch_val.append(metrics_val[2])

            values = [('val_loss', loss_val)]
            progBar.update((it + iv + 2) * batch_size, values=values)

        epoch_loss_val = np.average(loss_list_per_batch_val)
        val_loss.append(epoch_loss_val)

        epoch_ssim_val = np.average(ssim_list_per_batch_val)
        val_ssim.append(epoch_ssim_val)

        epoch_cpsnr_val = np.average(cpsnr_list_per_batch_val)
        val_cpsnr.append(epoch_cpsnr_val)

        loss_list_per_batch_val.clear()
        ssim_list_per_batch_val.clear()
        cpsnr_list_per_batch_val.clear()
        
        if scheduler == 1:

            print("patience_loop: ", patience_loop)
            print("epoch_loss_val: ", epoch_loss_val)
            print("loss: ", loss_patience)

            if epoch_loss_val >= loss_patience:
                patience_loop = patience_loop - 1
                if patience_loop == 0:
                    optimizer.lr.assign(optimizer.lr.read_value() * 0.1)
                    patience_loop = patience
            else:
                patience_loop = patience
                
            loss_patience = epoch_loss_val
                
        elif scheduler == 2:
            if epoch != 0 and epoch % 1 == 0:
                optimizer.lr.assign(optimizer.lr.read_value() * 0.1)
        else:
            pass

        str2 = "______ Training with learning rate: " + \
               str(optimizer.lr.read_value().numpy()) + "______\n"
        str3 = "Train Loss: " + str(epoch_loss_train) + "  CPSNR: " + \
               str(epoch_cpsnr_train) + "  SSIM: " + str(epoch_ssim_train) + "\n"
        str4 = "______Validation______\n"
        str5 = "Val Loss: " + str(epoch_loss_val) + "  CPSNR: " + str(epoch_cpsnr_val) \
               + "  SSIM: " + str(epoch_ssim_val) + "\n"
        
        print(str2)
        print(str3)
        print(str4)
        print(str5)
        
        with open(out_file, "a") as myfile:
            myfile.write(str1 + "/" + str(epochs) + ", Time needed(sec): " + str(int(time.time() - epoch_time_start)) + "\n")
            myfile.write(str2)
            myfile.write(str3)
            myfile.write(str4)
            myfile.write(str5)
            
            
            

        if (epoch + 1) % math.ceil(epochs / 4) == 0:
            model.save(dir_to_save + "_" + str(
                epoch + 1 + epoch_training) + "epochs.h5")
            np.savez((dir_to_save + "_" + str(
                epoch + 1 + epoch_training) + "epochs" + '_optimizer.npz'),
                    *optimizer.get_weights())
            
        if optimizer.lr.read_value().numpy() < 0.000000001:
            break
    
    model.save(dir_to_save + "_" + str(
                epoch + 1 + epoch_training) + "epochs.h5")
    np.savez((dir_to_save + "_" + str(
                epoch + 1 + epoch_training) + "epochs" + '_optimizer.npz'),
                    *optimizer.get_weights())
    

    return model, optimizer, np.array(val_loss), np.array(val_ssim), np.array(
        val_cpsnr), np.array(train_loss), np.array(
        train_ssim), np.array(train_cpsnr)


def prune(model, directory, mosaiced_directory, color_mode, start_sparsity,
          end_sparsity, batch_size=16, epochs=2,
          floating_point="float32"):
    # READ DATA

    with redirect_stdout(None):

        target_train_ds = image_dataset_from_directory(
            directory,
            floating_point,
            label_mode=None,
            validation_split=0.2,
            subset="training",
            seed=123,
            batch_size=batch_size,
            shuffle=False)

        input_train_ds = image_dataset_from_directory(
            mosaiced_directory,
            floating_point,
            label_mode=None,
            color_mode=color_mode,
            validation_split=0.2,
            subset="training",
            seed=123,
            batch_size=batch_size,
            shuffle=False)

        target_val_ds = image_dataset_from_directory(
            directory,
            floating_point,
            label_mode=None,
            validation_split=0.2,
            subset="validation",
            seed=123,
            batch_size=batch_size,
            shuffle=False)

        input_val_ds = image_dataset_from_directory(
            mosaiced_directory,
            floating_point,
            label_mode=None,
            color_mode=color_mode,
            validation_split=0.2,
            subset="validation",
            seed=123,
            batch_size=batch_size,
            shuffle=False)

    # PREPARE DATA
    target_train_ds = target_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    target_validation_ds = target_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    input_train_ds = input_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    input_validation_ds = input_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    validation_ds = tf.data.Dataset.zip(
        (target_validation_ds, input_validation_ds))
    train_ds = tf.data.Dataset.zip((target_train_ds, input_train_ds))

    # PREPARE VALUES FOR PROGBAR
    validation_samples = len(target_val_ds)
    training_samples = len(target_train_ds)

    num_samples = batch_size * (validation_samples + training_samples)

    metrics_names = ['train_loss', 'val_loss', 'CPSNR']

    # PREPARE LISTS FOR MONITORING LOSSES
    loss_list_per_batch = collections.deque()
    ssim_list_per_batch = collections.deque()
    cpsnr_list_per_batch = collections.deque()

    loss_list_per_batch_val = collections.deque()
    ssim_list_per_batch_val = collections.deque()
    cpsnr_list_per_batch_val = collections.deque()

    train_loss = []
    val_loss = []

    train_ssim = []
    val_ssim = []

    train_cpsnr = []
    val_cpsnr = []

    # PREPARE VALUES FOR OPTIMIZER AND LEARNING RATE SCHEDULER
    learning_rate = 0.0001
    optimizer = AdamW(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999,
                      weight_decay=1e-4)

    # PREPARE VALUES FOR PRUNING
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    end_step = np.ceil(training_samples).astype(np.int32) * epochs
    print("Initial_sparsity: ", start_sparsity / 100, "\nFinal_sparsity: ",
          end_sparsity / 100)
    pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=start_sparsity / 100,
        final_sparsity=end_sparsity / 100,
        begin_step=0,
        end_step=end_step,
        frequency=100)}

    model = prune_low_magnitude(model, **pruning_params)

    step_callback = tfmot.sparsity.keras.UpdatePruningStep()
    step_callback.set_model(model)
    step_callback.on_train_begin()

    for epoch in range(epochs):

        print("\nStart of epoch %d" % (epoch + 1))
        progBar = tf.keras.utils.Progbar(num_samples,
                                         stateful_metrics=metrics_names)

        # Training loop
        for it, (y_batch_train, x_batch_train) in enumerate(train_ds):
            step_callback.on_train_batch_begin(batch=-1)

            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                metrics_train = metrics_calc(y_batch_train, logits)
                loss = metrics_train[0]

            loss_list_per_batch.append(metrics_train[0])
            ssim_list_per_batch.append(metrics_train[1])
            cpsnr_list_per_batch.append(metrics_train[2])

            grads = tape.gradient(loss, model.trainable_weights)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            values = [('train_loss', metrics_train[0]),
                      ('CPSNR', metrics_train[2])]
            progBar.update((it + 1) * batch_size, values=values)

        epoch_loss_train = np.average(loss_list_per_batch)
        train_loss.append(epoch_loss_train)

        epoch_ssim_train = np.average(ssim_list_per_batch)
        train_ssim.append(epoch_ssim_train)

        epoch_cpsnr_train = np.average(cpsnr_list_per_batch)
        train_cpsnr.append(epoch_cpsnr_train)

        loss_list_per_batch.clear()
        ssim_list_per_batch.clear()
        cpsnr_list_per_batch.clear()

        # Validation loop
        for iv, (y_batch_val, x_batch_val) in enumerate(validation_ds):
            logits_val = model(x_batch_val)
            metrics_val = metrics_calc(y_batch_val, logits_val)

            loss_list_per_batch_val.append(metrics_val[0])
            ssim_list_per_batch_val.append(metrics_val[1])
            cpsnr_list_per_batch_val.append(metrics_val[2])

            values = [('val_loss', metrics_val[0])]
            progBar.update((it + iv + 2) * batch_size, values=values)

        step_callback.on_epoch_end(batch=-1)

        epoch_loss_val = np.average(loss_list_per_batch_val)
        val_loss.append(epoch_loss_val)

        epoch_ssim_val = np.average(ssim_list_per_batch_val)
        val_ssim.append(epoch_ssim_val)

        epoch_cpsnr_val = np.average(cpsnr_list_per_batch_val)
        val_cpsnr.append(epoch_cpsnr_val)

        loss_list_per_batch_val.clear()
        ssim_list_per_batch_val.clear()
        cpsnr_list_per_batch_val.clear()

        print("______ Training with learning rate: ", learning_rate,
              " ______\n")
        print("L2 Loss: ", epoch_loss_train, "  CPSNR: ", epoch_cpsnr_train,
              "  SSIM: ", epoch_ssim_train, "\n")
        print("______Validation______\n")
        print("L2 Loss: ", epoch_loss_val, "  CPSNR: ", epoch_cpsnr_val,
              "  SSIM: ", epoch_ssim_val, "\n")

    return model, np.array(val_loss), np.array(val_ssim), np.array(
        val_cpsnr), np.array(train_loss), np.array(
        train_ssim), np.array(train_cpsnr)


def testing_session(model, directory, mosaiced_directory, color_mode,
                    batch_size, floating_point="float32"):
    # READ DATA

    with redirect_stdout(None):
        target_test_ds = image_dataset_from_directory(
            directory,
            floating_point,
            label_mode=None,
            seed=123,
            batch_size=batch_size,
            shuffle=False)

        input_test_ds = image_dataset_from_directory(
            mosaiced_directory,
            floating_point,
            label_mode=None,
            color_mode=color_mode,
            seed=123,
            batch_size=batch_size,
            shuffle=False)

    # PREPARE DATA
    target_test_ds = target_test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    input_test_ds = input_test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.data.Dataset.zip((target_test_ds, input_test_ds))

    # PREPARE VALUES FOR PROGBAR
    testing_samples = len(target_test_ds)
    num_samples = batch_size * testing_samples

    metrics_names = ['test_loss', 'CPSNR']
    progBar = tf.keras.utils.Progbar(num_samples,
                                     stateful_metrics=metrics_names)

    # PREPARE LIST FOR MONITORING LOSS
    metrics = []

    for i, (y_batch_test, x_batch_test) in enumerate(test_ds):
        logits_test = model(x_batch_test)
        metrics_test = metrics_calc(y_batch_test, logits_test)

        metrics.append(metrics_test)
        values = [('test_loss', metrics_test[0]), ('CPSNR', metrics_test[2])]
        progBar.update(((i + 1) * batch_size), values=values)
    avg = np.average(metrics, axis=0)

    print("\n______Testing______\n")
    print("L2 Loss: ", avg[0], "  CPSNR: ", avg[2], "  SSIM: ", avg[1], "\n")


def inference_data(model, directory, cfa, channels, image_size, save_dir,
                   dataset, floating_point, tflite_model=False):
    flist_in = cfa_img.pics_in_dir(directory)

    if floating_point == "float16":
        cfa_img.fp_img = np.float16
    elif floating_point == "float32":
        cfa_img.fp_img = np.float32
    elif floating_point == "float64":
        cfa_img.fp_img = np.float64

    # for lite model set the interpreter
    if tflite_model:
        interpreter = tf.lite.Interpreter(model_path=str(model))
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

    metrics_dataset = []
    results = []
    for image in flist_in:
        img_cut = csp.cut_frame(image, shape=image_size)
        dtype = img_cut[0][0].dtype

        img_fp_org = []
        img_fp_dms = []
        for row in img_cut:
            input_img = [cfa_img.convert2fp(
                cfa_img.mosaic(img, cfa, order='RGGB', one_channel=channels))
                         for img in row]
            input_img = np.stack(input_img, axis=0)

            if tflite_model:
                demosaiced = []
                for img_in in input_img:
                    interpreter.set_tensor(input_index,
                                           np.expand_dims(img_in, axis=0))
                    interpreter.invoke()
                    demosaiced.append(interpreter.get_tensor(output_index))
                demosaiced = np.vstack(demosaiced)
            else:
                demosaiced = model(input_img)

            # keep original as float for metric calculations
            row = np.stack([cfa_img.convert2fp(img) for img in row], axis=0)
            img_fp_org.append(row)
            img_fp_dms.append(demosaiced.numpy())

        image = os.path.split(image)[1]
        fp_org = np.vstack([np.stack(img, axis=0) for img in img_fp_org])
        fp_dms = np.vstack([np.stack(img, axis=0) for img in img_fp_dms])
        metrics = [float(mm) for mm in metrics_calc(fp_org, fp_dms)]
        case = {'image': image, 'l2': metrics[0],
                'ssim': metrics[1], 'cpsnr': metrics[2]}
        print(case)

        dms = [[cfa_img.convert2int(img, dtype) for img in rw] for rw in
               img_fp_dms]
        fname, fext = os.path.splitext(image)
        csp.stitch_list(dms, os.path.join(save_dir, fname + '_stitched' + fext))

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
