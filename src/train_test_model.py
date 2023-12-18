from tensorflow_addons.optimizers import AdamW
import collections
import math
from contextlib import redirect_stdout
import cut_stitch_pics as csp
import cfa_image as cfa_img
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory
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
def cpsnr(original, contrast):
    mse = custom_mse_loss(original, contrast)
    if mse == 0:
        return tf.convert_to_tensor(100., dtype=mse.dtype)
    return -10. * tf.experimental.numpy.log10(mse)


@tf.function
def custom_mse_loss(y_true, y_pred):
    square_error = tf.math.square(y_true - y_pred)
    mean_square_error = tf.reduce_mean(square_error)
    return mean_square_error


@tf.function
def custom_l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.abs(y_true - y_pred))


@tf.function
def mix_ssim_l1_loss(y_true, y_pred):
    '''modified from tensorflow implementation'''
    # Convert to tensor if needed.
    img1 = tf.convert_to_tensor(y_true, name='img1')
    img2 = tf.convert_to_tensor(y_pred, name='img2')

    # SSIM parameters
    fsize = 11
    alpha_mix = 0.84
    max_val = tf.cast(1., tf.float32)
    filter_size = tf.constant(fsize, dtype=tf.int32)
    filter_sigma = tf.constant(1.5, dtype=tf.float32)
    k1 = 0.01
    k2 = 0.03
    compensation = 1.0

    img1 = tf.image.convert_image_dtype(img1, tf.float32)
    img2 = tf.image.convert_image_dtype(img2, tf.float32)

    # copied from _fspecial_gauss
    coords = tf.cast(tf.range(filter_size), tf.float32)
    coords -= tf.cast(filter_size - 1, tf.float32) / 2.0
    gfilt = tf.square(coords)
    gfilt *= -0.5 / tf.square(filter_sigma)
    gfilt = tf.reshape(gfilt, shape=[1, -1]) + tf.reshape(gfilt, shape=[-1, 1])
    gfilt = tf.reshape(gfilt, shape=[1, -1])  # For tf.nn.softmax().
    gfilt = tf.nn.softmax(gfilt)
    kernel = tf.reshape(gfilt, shape=[1, filter_size, filter_size, 1])
    kernel = tf.tile(kernel, multiples=[img1.shape[0], 1, 1, img1.shape[-1]])

    # image support for the kernel (at the center of square patch)
    pxc = img1.shape[1] // 2
    pxs = pxc - fsize // 2
    krn_sup = slice(pxs, pxs + fsize)
    img1s = img1[:, krn_sup, krn_sup, :]
    img2s = img2[:, krn_sup, krn_sup, :]
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    mean1 = tf.reduce_sum(img1s * kernel, axis=[1, 2])
    mean2 = tf.reduce_sum(img2s * kernel, axis=[1, 2])
    num0 = mean1 * mean2 * 2.0
    den0 = tf.square(mean1) + tf.square(mean2)
    luminance = (num0 + c1) / (den0 + c1)

    num1 = tf.reduce_sum(img1s * img2s * kernel, axis=[1, 2]) * 2.0
    den1 = tf.reduce_sum((tf.square(img1s) + tf.square(img2s)) * kernel,
                         axis=[1, 2])
    c2 *= compensation
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    ssim = tf.reduce_mean(luminance * cs, axis=[-1])

    l1 = kernel * tf.abs(img1s - img2s)
    l1 = tf.reduce_sum(l1, axis=[1, 2, 3])

    mix = (1. - ssim) * alpha_mix + (1. - alpha_mix) * l1
    return tf.reduce_mean(mix)


@tf.function
def metrics_calc(y_true, y_pred):
    l2_value = custom_mse_loss(y_true, y_pred)
    ssim_value = ssim_loss(y_true, y_pred)
    cpsnr_value = cpsnr(y_true, y_pred)
    l1_value = custom_l1_loss(y_true, y_pred)
    mix_ssim_l1 = mix_ssim_l1_loss(y_true, y_pred)
    return [l2_value, ssim_value, cpsnr_value, l1_value, mix_ssim_l1]


@tf.function
def train_step(model, x_batch_train, y_batch_train, optimizer, lid, lsc):
    with tf.GradientTape() as tape:
        logits = model(x_batch_train, training=True)
        metrics_train = metrics_calc(y_batch_train, logits)
        loss_train = lsc * metrics_train[lid]
        if np.abs(lsc) < 0.999999:
            loss_train += (1. + lsc) * metrics_train[3]

    grads = tape.gradient(loss_train, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return metrics_train, loss_train


def _dist_train_step(model, yx_batch_train, optimizer, global_bs, loss_object):
    with tf.GradientTape() as tape:
        logits = model(yx_batch_train[1], training=True)
        loss, mtrcs = loss_object(yx_batch_train[0], logits)
        loss_train = tf.nn.compute_average_loss(
            loss,
            global_batch_size=global_bs
        )

    # other metrics
    l2 = tf.nn.compute_average_loss(
        mtrcs[0],
        global_batch_size=global_bs
    )
    ssim = tf.nn.compute_average_loss(
        mtrcs[1],
        global_batch_size=global_bs
    )
    cpsnr = tf.nn.compute_average_loss(
        mtrcs[2],
        global_batch_size=global_bs
    )
    l1 = tf.nn.compute_average_loss(
        mtrcs[3],
        global_batch_size=global_bs
    )

    # optimize
    grads = tape.gradient(loss_train, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_train, [l2, ssim, cpsnr, l1]


def _dist_validation_step(model, yx_batch_val, global_bs, loss_object):
    logits = model(yx_batch_val[1])
    loss, mtrcs = loss_object(yx_batch_val[0], logits)
    loss_val = tf.nn.compute_average_loss(
        loss,
        global_batch_size=global_bs
    )
    l2 = tf.nn.compute_average_loss(
        mtrcs[0],
        global_batch_size=global_bs
    )
    ssim = tf.nn.compute_average_loss(
        mtrcs[1],
        global_batch_size=global_bs
    )
    cpsnr = tf.nn.compute_average_loss(
        mtrcs[2],
        global_batch_size=global_bs
    )
    l1 = tf.nn.compute_average_loss(
        mtrcs[3],
        global_batch_size=global_bs
    )
    return loss_val, [l2, ssim, cpsnr, l1]


@tf.function
def distributed_train_step(strategy, model, yx_batch_train, optimizer,
                           global_bs, loss_obj):
    train_args = (model, yx_batch_train, optimizer, global_bs, loss_obj)
    per_replica_losses = strategy.run(_dist_train_step, args=train_args)
    loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[0],
                           axis=None)
    metrics = []
    for imtr in per_replica_losses[1]:
        metrics.append(strategy.reduce(tf.distribute.ReduceOp.SUM, imtr,
                                       axis=None))
    return metrics, loss


@tf.function
def distributed_validation_step(strategy, model, yx_batch_val, global_bs,
                                loss_obj):
    val_args = (model, yx_batch_val, global_bs, loss_obj)
    per_replica_losses = strategy.run(_dist_validation_step, args=val_args)
    loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses[0],
                           axis=None)
    metrics = []
    for imtr in per_replica_losses[1]:
        metrics.append(strategy.reduce(tf.distribute.ReduceOp.SUM, imtr,
                                       axis=None))
    return metrics, loss


def _load_datasets(directory, mosaiced_directory,
                   image_size, color_mode, batch_size,
                   floating_point):
    # loading and transform parameters
    if floating_point == "float16":
        dtype = tf.dtypes.float16
    elif floating_point == "float32":
        dtype = tf.dtypes.float32
    elif floating_point == "float64":
        dtype = tf.dtypes.float64
    ndir = len(directory)
    img_sz = (image_size, image_size)

    # READ DATA
    fnull = open(os.devnull, 'w')
    train_ds = []
    validation_ds = []
    ntrain = 0
    nvalidation = 0
    with redirect_stdout(fnull):
        for idir, mos_dir in zip(directory, mosaiced_directory):
            targetds = image_dataset_from_directory(
                idir,
                labels=None,
                label_mode=None,
                image_size=img_sz,
                validation_split=0.2,
                subset="training",
                seed=123,
                batch_size=batch_size,
                shuffle=False)
            type_curr = targetds.element_spec.dtype
            scl_cnst = tf.constant(tf.dtypes.uint8.max, dtype=type_curr)
            targetds = targetds.map(
                lambda x: tf.cast(x / scl_cnst, dtype=dtype)
            )

            inputds = image_dataset_from_directory(
                mos_dir,
                labels=None,
                label_mode=None,
                image_size=img_sz,
                color_mode=color_mode,
                validation_split=0.2,
                subset="training",
                seed=123,
                batch_size=batch_size,
                shuffle=False)
            type_curr = inputds.element_spec.dtype
            scl_cnst = tf.constant(tf.dtypes.uint8.max, dtype=type_curr)
            inputds = inputds.map(
                lambda x: tf.cast(x / scl_cnst, dtype=dtype)
            )
            train_ds.append(tf.data.Dataset.zip((targetds, inputds)))
            ntrain += targetds.cardinality()

            targetds = image_dataset_from_directory(
                idir,
                labels=None,
                label_mode=None,
                image_size=img_sz,
                validation_split=0.2,
                subset="validation",
                seed=123,
                batch_size=batch_size,
                shuffle=False)
            type_curr = targetds.element_spec.dtype
            scl_cnst = tf.constant(tf.dtypes.uint8.max, dtype=type_curr)
            targetds = targetds.map(
                lambda x: tf.cast(x / scl_cnst, dtype=dtype)
            )

            inputds = image_dataset_from_directory(
                mos_dir,
                labels=None,
                label_mode=None,
                image_size=img_sz,
                color_mode=color_mode,
                validation_split=0.2,
                subset="validation",
                seed=123,
                batch_size=batch_size,
                shuffle=False)
            type_curr = inputds.element_spec.dtype
            scl_cnst = tf.constant(tf.dtypes.uint8.max, dtype=type_curr)
            inputds = inputds.map(
                lambda x: tf.cast(x / scl_cnst, dtype=dtype)
            )
            validation_ds.append(tf.data.Dataset.zip((targetds, inputds)))
            nvalidation += targetds.cardinality()

    # PREPARE DATA
    if ndir > 1:
        train_ds = tf.data.Dataset.sample_from_datasets(
            train_ds,
            seed=65444,
            stop_on_empty_dataset=False
        )
        validation_ds = tf.data.Dataset.sample_from_datasets(
            validation_ds,
            seed=65444,
            stop_on_empty_dataset=False
        )
    else:
        train_ds = train_ds[0]
        validation_ds = validation_ds[0]

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, validation_ds, (ntrain.numpy(), nvalidation.numpy())


def _select_metric(train_metric):
    if train_metric == 'l2':
        learning_rate = 0.0001
        mtrc_idx = 0
        mtrc_scl = 1.
    elif train_metric == 'ssim':
        learning_rate = 0.0001 * 10
        mtrc_idx = 1
        mtrc_scl = -.84
    elif train_metric == 'cpsnr':
        learning_rate = 0.0001
        mtrc_idx = 2
        mtrc_scl = -1.
    elif train_metric == 'ssim_l1_mix':
        learning_rate = 0.0001 * 10
        mtrc_idx = 4
    elif train_metric == 'cpsnr_l1_mix':
        learning_rate = 0.0001
        mtrc_idx = 2
        mtrc_scl = -.84
    else:
        msg = 'Unknown metric: ' + str(train_metric)
        raise ValueError(msg)

    return learning_rate, mtrc_idx, mtrc_scl


def training_session(model, directory, mosaiced_directory, color_mode,
                     image_size, scheduler, dir_to_save, batch_size=16,
                     epochs=5, floating_point="float32", patience=4,
                     train_metric="l2", continue_train=False):

    # READ AND PREPARE DATA
    train_ds, validation_ds, nbtch = _load_datasets(directory,
                                                    mosaiced_directory,
                                                    image_size, color_mode,
                                                    batch_size, floating_point)

    # CREATE FILE FOR LIVE TRACKING
    out_path = Path(dir_to_save)
    out_file = os.path.join(out_path.absolute(), 'train_live.txt')
    if not path.isfile(out_file):
        open(out_file, 'a').close()

    # PREPARE VALUES FOR PROGBAR
    training_samples, validation_samples = nbtch
    num_samples = batch_size * (validation_samples + training_samples)

    metrics_names = ['train_loss', 'val_loss', 'CPSNR', 'SSIM']

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

    learning_rate, mtrc_idx, mtrc_scl = _select_metric(train_metric)

    # PREPARE VALUES FOR OPTIMIZER AND LEARNING RATE SCHEDULER
    if scheduler == 2:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=4 * training_samples,
            decay_rate=0.1,
            staircase=True
        )
    # optimizer = AdamW(learning_rate=learning_rate, beta_1=0.9,
    #                   beta_2=0.999, weight_decay=1e-4)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.999)

    # optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    if scheduler == 1:
        schdlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.1,
                                                      patience=patience,
                                                      verbose=1,
                                                      mode='min',
                                                      min_delta=0.0001,
                                                      cooldown=3,
                                                      min_lr=1.e-12)
        model.optimizer = optimizer
        schdlr.model = model
        schdlr.on_train_begin()

    # create or load checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                               optimizer=optimizer,
                               model=model)
    ckpt_dir = os.path.join(dir_to_save, 'checkpoints', '')
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=4)
    cont_flag = continue_train is not None
    cont_flag = cont_flag and (manager.latest_checkpoint is not None)
    if cont_flag:
        ckpt.restore(manager.latest_checkpoint)
        print("Restored from {}".format(manager.latest_checkpoint))
        if continue_train > 0.:
            print("Resetting learning rate to {:.5g}".format(continue_train))
            optimizer.lr.assign(continue_train)
    else:
        print("Initializing from scratch.")
    epoch_start = int(ckpt.step)

    for epoch in range(epoch_start, epoch_start + epochs):
        epoch_time_start = time.time()
        str1 = "\nStart of epoch %d" % (epoch + 1)
        print(str1)

        ckpt.step.assign_add(1)
        progbar = tf.keras.utils.Progbar(num_samples,
                                         stateful_metrics=metrics_names)

        # Training loop
        for it, (y_batch_train, x_batch_train) in enumerate(train_ds):
            metrics_train, loss_train = train_step(model,
                                                   x_batch_train,
                                                   y_batch_train,
                                                   optimizer,
                                                   mtrc_idx,
                                                   mtrc_scl)

            loss_list_per_batch.append(loss_train)
            ssim_list_per_batch.append(metrics_train[1])
            cpsnr_list_per_batch.append(metrics_train[2])

            values = [('train_loss', loss_train),
                      ('CPSNR', metrics_train[2]),
                      ('SSIM', metrics_train[1])]
            progbar.update((it + 1) * batch_size, values=values)

        epoch_loss_train = np.average(loss_list_per_batch)
        train_loss.append(epoch_loss_train)

        epoch_ssim_train = np.average(ssim_list_per_batch)
        train_ssim.append(epoch_ssim_train)

        epoch_cpsnr_train = np.average(cpsnr_list_per_batch)
        train_cpsnr.append(epoch_cpsnr_train)

        loss_list_per_batch.clear()
        ssim_list_per_batch.clear()
        cpsnr_list_per_batch.clear()

        if int(ckpt.step) % 5 == 0:
            manager.save()

        # Validation loop
        for iv, (y_batch_val, x_batch_val) in enumerate(validation_ds):
            logits_val = model(x_batch_val)
            metrics_val = metrics_calc(y_batch_val, logits_val)
            loss_val = mtrc_scl * metrics_val[mtrc_idx]
            if np.abs(mtrc_scl) < 0.999999:
                loss_val += (1. + mtrc_scl) * metrics_val[3]

            loss_list_per_batch_val.append(loss_val)
            ssim_list_per_batch_val.append(metrics_val[1])
            cpsnr_list_per_batch_val.append(metrics_val[2])

            values = [('val_loss', loss_val)]
            progbar.update((it + iv + 2) * batch_size, values=values)

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
            schdlr.on_epoch_end(epoch, {'val_loss': val_loss[-1]})

        if scheduler == 2:
            lr_print = optimizer._decayed_lr(tf.float32).numpy()
        else:
            lr_print = optimizer.lr.numpy()

        str2 = "______Training with learning rate: " + \
            str(lr_print) + "______\n"
        str3 = "Train Loss: " + str(epoch_loss_train) + "  CPSNR: " + \
            str(epoch_cpsnr_train) + "  SSIM: " + str(epoch_ssim_train) + "\n"
        str4 = "______Validation______\n"
        str5 = "Val Loss: " + str(epoch_loss_val) + "  CPSNR: " + \
            str(epoch_cpsnr_val) + "  SSIM: " + str(epoch_ssim_val) + "\n"

        print(str2)
        print(str3)
        print(str4)
        print(str5)

        with open(out_file, "a") as myfile:
            myfile.write(str1 + "/" + str(epochs) + ", Time needed(sec): "
                         + str(int(time.time() - epoch_time_start)) + "\n")
            myfile.write(str2)
            myfile.write(str3)
            myfile.write(str4)
            myfile.write(str5)

        # FIXME: shouldn't old models be erased when called without continue training?
        if (epoch + 1) % math.ceil(epochs / 4) == 0:
            save_name = "model_" + str(epoch + 1) + "epochs.h5"
            save_path = os.path.join(dir_to_save, save_name)
            print(save_path)
            model.save(save_path)

        # termination criterion for training loop
        if optimizer.lr.read_value().numpy() < 0.000000001:
            break

    # save last iteration
    manager.save()

    return model, np.array(val_loss), np.array(val_ssim), \
        np.array(val_cpsnr),  np.array(train_loss), np.array(train_ssim), \
        np.array(train_cpsnr)


def training_session_distribute(strategy, model, directory, mosaiced_directory,
                                color_mode, image_size, scheduler,
                                dir_to_save, batch_size=16, epochs=5,
                                floating_point="float32", patience=4,
                                train_metric="l2", continue_train=False):

    # load the data sets
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    train_ds, validation_ds, nbtch = _load_datasets(directory,
                                                    mosaiced_directory,
                                                    image_size, color_mode,
                                                    global_batch_size,
                                                    floating_point)

    # do not warn about FILE vs DATA shard policy
    options = tf.data.Options()
    data_shard = tf.data.experimental.AutoShardPolicy.DATA
    options.experimental_distribute.auto_shard_policy = data_shard
    train_ds = train_ds.with_options(options)
    val_ds = validation_ds.with_options(options)

    train_dist_ds = strategy.experimental_distribute_dataset(train_ds)
    val_dist_ds = strategy.experimental_distribute_dataset(val_ds)

    # PREPARE VALUES FOR PROGBAR
    training_samples, validation_samples = nbtch
    num_samples = global_batch_size * (validation_samples + training_samples)

    # set training parameters here
    learning_rate, mtrc_idx, mtrc_scl = _select_metric(train_metric)

    # create file for live tracking
    out_path = Path(dir_to_save)
    out_file = os.path.join(out_path.absolute(), 'train_live.txt')
    if not path.isfile(out_file):
        open(out_file, 'a').close()

    with strategy.scope():
        def metrics_calc_dist(y_true, y_pred):
            axm = (1, 2, 3)
            dlog = tf.constant(1.e-45, dtype=np.float32)
            l1 = tf.reduce_mean(tf.math.abs(y_true - y_pred), axis=axm)
            l2 = tf.reduce_mean(tf.math.square(y_true - y_pred), axis=axm)
            ssim = tf.image.ssim(y_true, y_pred, 1.)
            cpsnr = -10. * tf.experimental.numpy.log10(l2 + dlog)
            mtrc = [l2, ssim, cpsnr, l1]
            loss = mtrc_scl * mtrc[mtrc_idx]
            if np.abs(mtrc_scl) < 0.999999:
                loss += (1. + mtrc_scl) * mtrc[3]
            return loss, mtrc

        # TODO: set the decay steps, somehow
        if scheduler == 2:
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=4 * training_samples,
                decay_rate=0.1,
                staircase=True
            )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             beta_1=0.9,
                                             beta_2=0.999)

    if scheduler == 1:
        schdlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.1,
                                                      patience=patience,
                                                      verbose=1,
                                                      mode='min',
                                                      min_delta=0.0001,
                                                      cooldown=3,
                                                      min_lr=1.e-12)
        model.optimizer = optimizer
        schdlr.model = model
        schdlr.on_train_begin()

    # checkpoints
    ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                               optimizer=optimizer,
                               model=model)
    ckpt_dir = os.path.join(dir_to_save, 'checkpoints', '')
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=4)
    cont_flag = continue_train is not None
    cont_flag = cont_flag and (manager.latest_checkpoint is not None)
    if cont_flag:
        ckpt.restore(manager.latest_checkpoint)
        print("Restored from {}".format(manager.latest_checkpoint))
        if continue_train > 0.:
            print("Resetting learning rate to {:.5g}".format(continue_train))
            optimizer.lr.assign(continue_train)
    else:
        print("Initializing from scratch.")
    epoch_start = int(ckpt.step.numpy())

    epoch_start = 0
    train_loss = []
    val_loss = []
    train_ssim = []
    val_ssim = []
    train_cpsnr = []
    val_cpsnr = []
    metrics_names = ['train_loss', 'val_loss', 'CPSNR', 'SSIM']
    for epoch in range(epoch_start, epoch_start + epochs):
        epoch_time_start = time.time()
        str1 = "\nStart of epoch %d" % (epoch + 1)
        print(str1)
        ckpt.step.assign_add(1)
        progbar = tf.keras.utils.Progbar(num_samples,
                                         stateful_metrics=metrics_names)

        # Training loop
        epoch_loss_train = 0.
        epoch_cpsnr_train = 0.
        epoch_ssim_train = 0.
        num_batches = 0
        for it, yx_batch_train in enumerate(train_dist_ds):
            metrics_train, loss_train = distributed_train_step(
                strategy, model, yx_batch_train, optimizer,
                global_batch_size,  metrics_calc_dist
            )
            epoch_loss_train += loss_train.numpy()
            epoch_ssim_train += metrics_train[1].numpy()
            epoch_cpsnr_train += metrics_train[2].numpy()
            num_batches += 1

            values = [('train_loss', loss_train.numpy()),
                      ('CPSNR', metrics_train[2].numpy()),
                      ('SSIM', metrics_train[1].numpy())]
            progbar.update((it + 1) * global_batch_size, values=values)
        epoch_loss_train /= float(num_batches)
        epoch_cpsnr_train /= float(num_batches)
        epoch_ssim_train /= float(num_batches)
        train_loss.append(epoch_loss_train)
        train_cpsnr.append(epoch_cpsnr_train)
        train_ssim.append(epoch_ssim_train)

        if int(ckpt.step.numpy()) % 5 == 0:
            manager.save()

        # Validation loop
        epoch_loss_val = 0.
        epoch_cpsnr_val = 0.
        epoch_ssim_val = 0.
        num_batches = 0
        for iv, yx_batch_val in enumerate(val_dist_ds):
            metrics_val, loss_val = distributed_validation_step(
                strategy, model, yx_batch_val,
                global_batch_size, metrics_calc_dist
            )
            epoch_loss_val += loss_val.numpy()
            epoch_ssim_val += metrics_val[1].numpy()
            epoch_cpsnr_val += metrics_val[2].numpy()
            num_batches += 1

            values = [('val_loss', loss_val.numpy())]
            progbar.update((it + iv + 2) * global_batch_size, values=values)
        epoch_loss_val /= num_batches
        epoch_cpsnr_val /= num_batches
        epoch_ssim_val /= num_batches
        val_loss.append(epoch_loss_val)
        val_cpsnr.append(epoch_cpsnr_val)
        val_ssim.append(epoch_ssim_val)

        if scheduler == 1:
            schdlr.on_epoch_end(epoch, {'val_loss': val_loss[-1]})

        if scheduler == 2:
            lr_print = optimizer._decayed_lr(tf.float32).numpy()
        else:
            lr_print = optimizer.lr.numpy()

        str2 = "______Training with learning rate: {:.6g}______\n"
        str2 = str2.format(lr_print)
        str3 = "Train Loss: {:.6g}  CPSNR: {:6g}  SSIM: {:6g}\n"
        str3 = str3.format(epoch_loss_train, epoch_cpsnr_train,
                           epoch_ssim_train)
        str4 = "______Validation______\n"
        str5 = "Val Loss: {:.6g}  CPSNR: {:.6g}  SSIM: {:.6g}\n"
        str5 = str5.format(epoch_loss_val, epoch_cpsnr_val, epoch_ssim_val)

        print(str2)
        print(str3)
        print(str4)
        print(str5)

        with open(out_file, "a") as myfile:
            myfile.write(str1 + "/" + str(epochs) + ", Time needed(sec): "
                         + str(int(time.time() - epoch_time_start)) + "\n")
            myfile.write(str2)
            myfile.write(str3)
            myfile.write(str4)
            myfile.write(str5)

    # save last iteration
    manager.save()

    return model, np.array(val_loss), np.array(val_ssim), \
        np.array(val_cpsnr),  np.array(train_loss), np.array(train_ssim), \
        np.array(train_cpsnr)


def prune(model, directory, mosaiced_directory, color_mode, start_sparsity,
          end_sparsity, batch_size=16, epochs=2, floating_point="float32"):

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

    validation_ds = tf.data.Dataset.zip((target_validation_ds, input_validation_ds))
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
    optimizer = AdamW(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, weight_decay=1e-4)

    # PREPARE VALUES FOR PRUNING
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    end_step = np.ceil(training_samples).astype(np.int32) * epochs
    print("Initial_sparsity: ", start_sparsity / 100, "\nFinal_sparsity: ", end_sparsity / 100)
    pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=start_sparsity / 100,
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
        progBar = tf.keras.utils.Progbar(num_samples, stateful_metrics=metrics_names)

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
            values = [('train_loss', metrics_train[0]), ('CPSNR', metrics_train[2])]
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

        print("______ Training with learning rate: ", learning_rate, " ______\n")
        print("L2 Loss: ", epoch_loss_train, "  CPSNR: ", epoch_cpsnr_train, "  SSIM: ", epoch_ssim_train, "\n")
        print("______Validation______\n")
        print("L2 Loss: ", epoch_loss_val, "  CPSNR: ", epoch_cpsnr_val, "  SSIM: ", epoch_ssim_val, "\n")

    return model, np.array(val_loss), np.array(val_ssim), np.array(val_cpsnr), np.array(train_loss), np.array(
        train_ssim), np.array(train_cpsnr)


def testing_session(model, directory, mosaiced_directory, color_mode, batch_size, floating_point = "float32"):

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
    progBar = tf.keras.utils.Progbar(num_samples, stateful_metrics=metrics_names)

    # PREPARE LIST FOR MONITORING LOSS
    metrics = []

    for i, (y_batch_test, x_batch_test) in enumerate(test_ds):

        logits_test = model(x_batch_test)
        metrics_test = metrics_calc(y_batch_test, logits_test)

        metrics.append(metrics_test)
        values = [('test_loss', metrics_test[0]), ('CPSNR', metrics_test[2])]
        progBar.update(((i+1) * batch_size), values=values)
    avg = np.average(metrics, axis=0)

    print("\n______Testing______\n")
    print("L2 Loss: ", avg[0], "  CPSNR: ", avg[2], "  SSIM: ", avg[1], "\n")


def inference_data(model, directory, cfa, channels, image_size, save_dir,
                   dataset, floating_point, order, tflite_model=False):
    flist_in = cfa_img.pics_in_dir(directory)

    ordering = order.upper()
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
            input_img = []
            for img in row:
                tmp_img = cfa_img.mosaic(img, cfa, order=ordering,
                                         one_channel=channels)
                input_img.append(cfa_img.convert2fp(tmp_img))
            input_img = np.stack(input_img, axis=0)

            if tflite_model:
                demosaiced = []
                for img_in in input_img:
                    interpreter.set_tensor(input_index, np.expand_dims(img_in,
                                                                       axis=0))
                    interpreter.invoke()
                    demosaiced.append(interpreter.get_tensor(output_index))
                demosaiced = np.vstack(demosaiced)
            else:
                demosaiced = model(input_img).numpy()

            # keep original as float for metric calculations
            row = np.stack([cfa_img.convert2fp(img) for img in row], axis=0)
            img_fp_org.append(row)
            img_fp_dms.append(demosaiced)

        image = os.path.split(image)[1]
        fp_org = np.vstack([np.stack(img, axis=0) for img in img_fp_org])
        fp_dms = np.vstack([np.stack(img, axis=0) for img in img_fp_dms])
        metrics = [float(mm) for mm in metrics_calc(fp_org, fp_dms)]
        case = {'image': image, 'l2': metrics[0],
                'ssim': metrics[1], 'cpsnr': metrics[2]}
        print(case)

        dms = [[cfa_img.convert2int(img, dtype) for img in rw] for rw in img_fp_dms]
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
