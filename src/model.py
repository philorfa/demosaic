import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Lambda
from keras.layers import Conv2D, Conv2DTranspose, Add, PReLU, ReLU, Concatenate, Layer,Subtract
import tensorflow_model_optimization as tfmot
from cfa_image import cfa_mask
import numpy as np

'''
class Decompose1D(Layer):

    def __init__(self, cfa, **kwargs):
        super(Decompose1D, self).__init__(**kwargs)
        self.cfa = cfa
        
    def build(self, input_shape):
        r = tf.Variable(initial_value=cfa_mask(np.ones((input_shape[1], input_shape[2], 3)), cfa=self.cfa)[:, :, 0],
                          trainable=False)
        self.r = tf.expand_dims(r, -1).numpy()

        g = tf.Variable(initial_value=cfa_mask(np.ones((input_shape[1], input_shape[2], 3)), cfa=self.cfa)[:, :, 1],
                            trainable=False)
        self.g = tf.expand_dims(g, -1).numpy()

        b = tf.Variable(initial_value=cfa_mask(np.ones((input_shape[1], input_shape[2], 3)), cfa=self.cfa)[:, :, 2],
                           trainable=False)
        self.b = tf.expand_dims(b, -1).numpy()
        
    def get_config(self):
        config = super(Decompose1D, self).get_config()
        config.update({'cfa': self.cfa})
        return config

    def call(self, inputs):  # Defines the computation from inputs to outputs
        r = tf.math.multiply(inputs, self.r)
        g = tf.math.multiply(inputs, self.g)
        b = tf.math.multiply(inputs, self.b)
        rgb = PrunableConcat(axis=3)([r, g, b])
        return rgb '''
class Decompose1D(Layer):

    def __init__(self, cfa, **kwargs):
        super(Decompose1D, self).__init__(**kwargs)
        self.cfa = cfa
        
    def build(self, input_shape):
        ones = np.ones((input_shape[1], input_shape[2], 3))
        msk = cfa_mask(ones, cfa=self.cfa).astype(self.dtype)
        self.cfa_mask = tf.Variable(initial_value=msk, trainable=False)
        
    def get_config(self):
        config = super(Decompose1D, self).get_config()
        config.update({'cfa': self.cfa})
        return config

    def call(self, inputs):  # Defines the computation from inputs to outputs
        return tf.math.multiply(inputs, self.cfa_mask)

class Clip(Layer):

    def __init__(self, min_value, max_value, **kwargs):
        super(Clip, self).__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        
    def get_config(self):
        config = super(Clip, self).get_config()
        config.update({'min_value': self.min_value})
        config.update({'max_value': self.max_value})
        return config

    def call(self, inputs, training=None):  # Defines the computation from inputs to outputs
        return inputs if training else tf.clip_by_value(inputs, clip_value_min=self.min_value, clip_value_max=self.max_value)
        


class PrunablePReLu(PReLU, tfmot.sparsity.keras.PrunableLayer):
    
    # f(x) = alpha * x for x < 0
    # f(x) = x for x >= 0
    def get_prunable_weights(self):
        return [self.alpha]


class PrunableConcat(Concatenate, tfmot.sparsity.keras.PrunableLayer):
    
    # zero parameters/weights
    def get_prunable_weights(self):
        return []
    
class PrunableDecompose1D(Decompose1D, tfmot.sparsity.keras.PrunableLayer):
    
    # zero parameters/weights
    def get_prunable_weights(self):
        return []
    
class PrunableClip(Clip, tfmot.sparsity.keras.PrunableLayer):
    
    # zero parameters/weights
    def get_prunable_weights(self):
        return []

def downscale(x_in, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same', strides=2)(x_in)
    x = ReLU()(x)
    return x


def upscale(x_in, num_filters):
    x = Conv2DTranspose(num_filters, kernel_size=2, padding='same', strides=2)(x_in)
    x = ReLU()(x)
    return x


def concat(x1, x2):
    x = PrunableConcat(axis=3)([x1, x2])
    return x


def feature_mapping(x_in, num_filters, function, bias_initializer):
    x = Conv2D(num_filters, kernel_size=3, padding='same', strides=1, bias_initializer=bias_initializer)(x_in)
    x1 = function()(x)
    x2 = Conv2D(num_filters, kernel_size=3, padding='same', strides=1, bias_initializer=bias_initializer)(x1)
    x3 = Add()([x2, x1])
    x4 = function()(x3)
    return x4


def feature_growth(resolution_level, alpha, beta):
    a = alpha
    b = beta
    y = a * resolution_level + b
    return y


def duplex_3ch(input_size=128, 
               pyramid_level=4, 
               function=0, 
               alpha=16, 
               beta=32, 
               model_policy = "float32", 
               stride = False, 
               bias = 0):
    
    bias_initializer = 'zero' if bias == 0 else 'glorot_uniform'
    
    tf.keras.mixed_precision.set_global_policy(model_policy)
        
    # recommended values
    # input_size = 128, pyramid_level = 4
    # input_size = 64, pyramid_level = 3
    
    activation = [PrunablePReLu, ReLU]
    size = (input_size, input_size, 3)
    input = Input(size)

    # pyramid up, and save output for skip connections
    
    xft_list = []
    xds = input
    
    starting_level = 0
    
    if stride:
        
        xft = feature_mapping(xds, feature_growth(0, alpha, beta), activation[function], bias_initializer)
        x = Conv2D(feature_growth(0, alpha, beta), kernel_size=5, padding='same', strides=4)(xft)
        xds = ReLU()(x)
        xft_list.append(xft)
        
        starting_level = 1
    
    
    for growth_level in range(starting_level, pyramid_level + 1):
        xft = feature_mapping(xds, feature_growth(growth_level, alpha, beta), activation[function], bias_initializer)
        xds = downscale(xft, feature_growth(growth_level, alpha, beta))
        xft_list.append(xft)

    # pyramid down
    
    yprev = xds
    for growth_level in range(pyramid_level - 1, -1, -1):
        yus = upscale(yprev, feature_growth(growth_level, alpha, beta))
        yc = concat(yus, xft_list[growth_level + 1])
        yprev = feature_mapping(yc, feature_growth(growth_level, alpha, beta), activation[function], bias_initializer)

    
    # convert tensor to 3 channel output
    
    if stride:
        yus = Conv2DTranspose(3, kernel_size=4, padding='same', strides=4)(yprev)
        yus = ReLU()(yus)
    else:
        yus = upscale(yprev, 3)
    
    yc = concat(yus, xft_list[0])
    yprev = feature_mapping(yc, 3, activation[function], bias_initializer)

    out = Add()([input, yprev])
    
    out_clipped = PrunableClip(0., 1.)(out)
    
    model = keras.models.Model(input, out_clipped)

    return model


def duplex_1ch(input_size=128, 
               cfa="chame", 
               pyramid_level=4, 
               function=0, 
               alpha=16, 
               beta=32, 
               model_policy="float32",
               stride=False, 
               decompose =False,
               bias = 0):
    
    bias_initializer = 'zero' if bias == 0 else 'glorot_uniform'
    
    tf.keras.mixed_precision.set_global_policy(model_policy)
   
    # recommended values
    # input_size = 128, pyramid_level = 4
    # input_size = 64, pyramid_level = 3
    
    activation = [PrunablePReLu, ReLU]
    size = (input_size, input_size, 1)
    input = Input(size)

    # pyramid up, and save output for skip connections
    xft_list = []
    xds = input
    
    starting_level = 0
    
    if stride:
        
        xft = feature_mapping(xds, feature_growth(0, alpha, beta), activation[function], bias_initializer)
        x = Conv2D(feature_growth(0, alpha, beta), kernel_size=5, padding='same', strides=4)(xft)
        xds = ReLU()(x)
        xft_list.append(xft)
        
        starting_level = 1
    
    
    for growth_level in range(starting_level, pyramid_level + 1):
        xft = feature_mapping(xds, feature_growth(growth_level, alpha, beta), activation[function], bias_initializer)
        xds = downscale(xft, feature_growth(growth_level, alpha, beta))
        xft_list.append(xft)

    # pyramid down
    yprev = xds
    for growth_level in range(pyramid_level - 1, -1, -1):
        yus = upscale(yprev, feature_growth(growth_level, alpha, beta))
        yc = concat(yus, xft_list[growth_level + 1])
        yprev = feature_mapping(yc, feature_growth(growth_level, alpha, beta), activation[function], bias_initializer)

    # convert tensor to 3 channel output
    
    if stride:
        yus = Conv2DTranspose(3, kernel_size=4, padding='same', strides=4)(yprev)
        yus = ReLU()(yus)
    else:
        yus = upscale(yprev, 3)
        
    yc = concat(yus, xft_list[0])
    yprev = feature_mapping(yc, 3, activation[function], bias_initializer)

    # 1 channel -> 3 channels
    decomp = PrunableDecompose1D(cfa)(input)

    out = Add()([yprev, decomp])
    
    out_clipped = PrunableClip(min_value=0., max_value=1.)(yprev)
    if decompose:
        out_clipped = PrunableClip(min_value=0., max_value=1.)(out)
    model = keras.models.Model(input, out_clipped)

    return model