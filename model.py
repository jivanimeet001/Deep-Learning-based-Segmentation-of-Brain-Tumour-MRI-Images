
import numpy as np
from keras.models import Model
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D
from keras.layers import  GaussianNoise, Input, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2DTranspose, UpSampling2D, concatenate, add
from keras.optimizers import SGD
import keras.backend as K
from losses import *

# set image data format to "channels_last"
K.set_image_data_format("channels_last")


class Unet_model(object):
    # Initialize class with image shape and optionally model weights
    def __init__(self, img_shape, load_model_weights=None):
        self.img_shape = img_shape
        self.load_model_weights = load_model_weights
        # Compile model
        self.model = self.compile_unet()

    # Define a function called compile_unet that returns a compiled U-Net model
    def compile_unet(self):
        # Define an input tensor with specified input shape
        i = Input(shape=self.img_shape)
        # Add a Gaussian noise layer to input tensor
        i_ = GaussianNoise(0.01)(i)
        # Add a convolutional layer with 64 filters, kernel size 2x2, same padding, and channel-last data format to input tensor
        i_ = Conv2D(64, 2, padding='same', data_format='channels_last')(i_)
        # Get output tensor of U-Net architecture by passing input tensor through unet() function
        out = self.unet(inputs=i_)
        # Define a Keras model with input i and output out
        model = Model(input=i, output=out)
        # Define a stochastic gradient descent optimizer with a learning rate of 0.08, momentum of 0.9, decay rate of 5e-6, and Nesterov momentum disabled
        sgd = SGD(lr=0.08, momentum=0.9, decay=5e-6, nesterov=False)
        # Compile model with Dice loss function and specified optimizer, and calculate metrics for whole tumor, core region, and enhancing region
        model.compile(loss=generate_dice_loss, optimizer=sgd, metrics=[
                      dice_whole_tumor_metric, dice_core_region_metric, dice_enhancingancing_metric])
        # If specified, load weights of trained model
        if self.load_model_weights is not None:
            model.load_weights(self.load_model_weights)

        # Return compiled model
        return model

    def unet(self, inputs, nb_classes=4, start_ch=64, depth=3, inc_rate=2., activation='relu', dropout=0.0, batchnorm=True, upconv=True, format_='channels_last'):

        # Define U-Net architecture using level_block function
        o = self.level_block(inputs, start_ch, depth, inc_rate,
                             activation, dropout, batchnorm, upconv, format_)

        # Apply Batch Normalization to output of last convolutional layer
        o = BatchNormalization()(o)

        # Apply PReLU activation function to output of Batch Normalization layer
        o = PReLU(shared_axes=[1, 2])(o)

        # Apply a 1x1 convolutional layer to reduce number of channels to number of classes
        o = Conv2D(nb_classes, 1, padding='same', data_format=format_)(o)

        # Apply softmax activation function to get class probabilities
        o = Activation('softmax')(o)

        # Return output
        return o


# A function that creates a U-Net level block
# m: input tensor
# dim: dimensionality of output space
# depth: depth of level block
# inc: rate of increasing output space dimensionality
# acti: activation function to use
# do: dropout rate
# bn: whether to use batch normalization
# up: whether to use up-sampling or transpose convolution
# format_: image data format

    def level_block(self, inputs, filters, depth, inc_rate, activation, dropout, batchnorm, upconv, format_='channels_last'):
        if depth > 0:
            # Encoding path
            x = self.encoder_residual_block(inputs, dropout, filters, activation, batchnorm, format_)
            # Down-sampling
            inputs = Conv2D(int(inc_rate*filters), 2, strides=2, padding='same', data_format=format_)(x)
            # Recursive call to next depth level
            inputs = self.level_block(inputs, int(inc_rate*filters), depth-1, inc_rate, activation, dropout, batchnorm, upconv, format_)
            # Up-sampling or transpose convolution
            if upconv:
                inputs = UpSampling2D(size=(2, 2), data_format=format_)(inputs)
                inputs = Conv2D(filters, 2, padding='same', data_format=format_)(inputs)
            else:
                inputs = Conv2DTranspose(filters, 3, strides=2, padding='same', data_format=format_)(inputs)
            # Concatenating output of down-sampling and up-sampling paths
            x = concatenate([x, inputs])
            # Decoding path
            inputs = self.decoder_residual_block(x, dropout, filters, activation, batchnorm, format_)
        else:
            # Single block
            inputs = self.encoder_residual_block(inputs, dropout, filters, activation, batchnorm, format_)
        return inputs

    def encoder_residual_block(inputs, dropout_rate, filters, activation, use_batchnorm, data_format="channels_last"):
        # Defines a residual block for encoder part of network
        x = inputs
        # Applies batch normalization to input tensor
        if use_batchnorm:
            x = BatchNormalization()(x)
        # Applies PReLU activation function with shared axes to input tensor
        x = PReLU(shared_axes=[1, 2])(x)
        # Applies a 3x3 convolution to input tensor with specified number of filters
        x = Conv2D(filters, 3, padding='same', data_format=data_format)(x)
        # Applies batch normalization to output of previous convolution
        if use_batchnorm:
            x = BatchNormalization()(x)
        # Applies PReLU activation function with shared axes to output of previous batch normalization
        x = PReLU(shared_axes=[1, 2])(x)
        # Applies a second 3x3 convolution to output of previous activation function with specified number of filters
        x = Conv2D(filters, 3, padding='same', data_format=data_format)(x)
        # Adds input tensor to output of second convolution (residual connection)
        x = add([inputs, x])
        return x

    def decoder_residual_block(inputs, dropout_rate, filters, activation, use_batchnorm, data_format="channels_last"):
        # Defines a residual block for decoder part of network
        x = inputs
        # Applies batch normalization to input tensor
        if use_batchnorm:
            x = BatchNormalization()(x)
        # Applies PReLU activation function with shared axes to input tensor
        x = PReLU(shared_axes=[1, 2])(x)
        # Applies a 3x3 convolution to input tensor with specified number of filters
        x = Conv2D(filters, 3, padding='same', data_format=data_format)(x)
        # Applies batch normalization to output of previous convolution
        if use_batchnorm:
            x = BatchNormalization()(x)
        # Applies PReLU activation function with shared axes to output of previous batch normalization
        x = PReLU(shared_axes=[1, 2])(x)
        # Applies a second 3x3 convolution to output of previous activation function with specified number of filters
        x = Conv2D(filters, 3, padding='same', data_format=data_format)(x)
        # Applies a 1x1 convolution to input tensor with same number of filters as output tensor (skip connection)
        skip = Conv2D(filters, 1, padding='same',
                      data_format=data_format, use_bias=False)(inputs)
        # Adds skip connection to output of second convolution (residual connection)
        x = add([skip, x])
        return x
