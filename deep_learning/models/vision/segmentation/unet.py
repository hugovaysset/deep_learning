import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model, layers
import tensorflow.keras.backend as K

class Unet:

    def conv2d_block(input_tensor, n_filters, kernel_size=(3, 3), dropout=None,
                 activation="relu", kernel_regul=None):
        """
        Function to add 2 convolutional layers with the parameters passed to it
        """
        # first layer
        if kernel_regul is None:
            x = layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                            kernel_initializer='he_normal', padding='same')(input_tensor)
        else:
            x = layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                            kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(kernel_regul),
                            use_bias=False)(input_tensor)
        if dropout:
            x = layers.Dropout(0.2)(x)
        if activation == "relu" or activation == "sigmoid" or activation == "linear":
            x = layers.Activation(activation)(x)
        elif activation == "leaky_relu" or activation == "l_relu" or activation == "lrelu":
            x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        else:
            raise NotImplementedError("activation function should be given by a valid string of leaky_relu")

        # second layer
        if kernel_regul is None:
            x = layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                            kernel_initializer='he_normal', padding='same', kernel_regularizer=kernel_regul)(x)
        else:
            x = layers.Conv2D(filters=n_filters, kernel_size=kernel_size,
                            kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(kernel_regul),
                            use_bias=False)(x)
        if dropout:
            x = layers.Dropout(0.2)(x)
        if activation == "relu" or activation == "sigmoid" or activation == "linear":
            x = layers.Activation(activation)(x)
        elif activation == "leaky_relu":
            x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        else:
            raise NotImplementedError("activation function should be given by a valid string of leaky_relu")

        return x

    def get_model(nb_filters, input_shape, output_channels, kernel_size=(3, 3), n_contractions=4, final_activation="sigmoid"):
        """
        nb_filters (int): kernel side
        input_shape (tuple): shape of one input image (x, y, c)
        n_contractions (int): number of contraction blocks (4 by defautl as in U-Net paper)  # TODO add this
        final_activation (str): any activation taken by keras
        kernel_size (tuple): size of the convolution kernels
        """
        n_channels_imgs, n_channels_masks = input_shape[-1], output_channels
        print(f"# input channels : {n_channels_imgs}.")
        print(f"# output channels : {n_channels_masks}.")

        initializer = initializer = tf.keras.initializers.GlorotNormal()
        entree = layers.Input(shape=(input_shape[0], input_shape[1], n_channels_imgs), dtype='float16')

        # contraction block 1
        result = layers.Conv2D(nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(entree)
        result = layers.BatchNormalization()(result)
        result = layers.Conv2D(nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(result)
        result1 = layers.BatchNormalization()(result)

        result = layers.MaxPool2D()(result1)

        # contraction block 2
        result = layers.Conv2D(2*nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(result)
        result = layers.BatchNormalization()(result)
        result = layers.Conv2D(2*nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(result)
        result2 = layers.BatchNormalization()(result)

        result = layers.MaxPool2D()(result2)

        # contraction block 3
        result = layers.Conv2D(4*nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(result)
        result = layers.BatchNormalization()(result)
        result = layers.Conv2D(4*nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(result)
        result3 = layers.BatchNormalization()(result)

        result = layers.MaxPool2D()(result3)

        # contraction block 4
        result = layers.Conv2D(4*nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(result)
        result = layers.BatchNormalization()(result)
        result = layers.Conv2D(4*nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(result)
        result4 = layers.BatchNormalization()(result)

        result = layers.MaxPool2D()(result4)

        # upsampling block 1
        result = layers.Conv2D(8*nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(result)
        result = layers.BatchNormalization()(result)
        result = layers.Conv2D(4*nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(result)
        result = layers.BatchNormalization()(result)

        result = layers.UpSampling2D()(result)
        result = tf.concat([result, result4], axis=-1)

        # upsampling block 2
        result = layers.Conv2D(8*nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(result)
        result = layers.BatchNormalization()(result)
        result = layers.Conv2D(4*nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(result)
        result = layers.BatchNormalization()(result)

        result = layers.UpSampling2D()(result)
        result = tf.concat([result, result3], axis=-1)

        # upsampling block 3
        result = layers.Conv2D(4*nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(result)
        result = layers.BatchNormalization()(result)
        result = layers.Conv2D(2*nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(result)
        result = layers.BatchNormalization()(result)

        result=layers.UpSampling2D()(result)
        result=tf.concat([result, result2], axis=-1)

        # upsampling block 4
        result = layers.Conv2D(2*nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(result)
        result = layers.BatchNormalization()(result)
        result = layers.Conv2D(nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(result)
        result = layers.BatchNormalization()(result)

        result = layers.UpSampling2D()(result)
        result = tf.concat([result, result1], axis=-1)

        # prediction block
        result = layers.Conv2D(nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(result)
        result = layers.BatchNormalization()(result)
        result = layers.Conv2D(nb_filters, kernel_size, activation='relu', padding='same', kernel_initializer=initializer)(result)
        result = layers.BatchNormalization()(result)

        sortie = layers.Conv2D(n_channels_masks, 1, activation=final_activation, padding='same', kernel_initializer=initializer)(result)

        model = Model(inputs=entree, outputs=sortie)
        return model
