import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D
from keras.layers import ConvLSTM2D, TimeDistributed,  merge, Conv2D
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Convolution2D, UpSampling2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate,Add
from keras.callbacks import ModelCheckpoint
import numpy as np
import keras.optimizers


############################################################################################
"""
Unet 3D.
"""
############################################################################################


def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, deconvolution=False,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics='mse',
                  batch_normalization=False, activation_name="sigmoid"):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

            current_layer = create_convolution_block(input_layer=current_layer, n_filters=256,
                                                   batch_normalization=batch_normalization)

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)

        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=4)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[4],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[4],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    # if not isinstance(metrics, list):
    #     metrics = [metrics]

    return model


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """

    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)




############################################################################################
"""
LSTM made in Leo.
"""
############################################################################################

def LSTMNET(input_shape):
    c = 12
    input_img = Input(input_shape, name='input')
    x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(input_img)
    x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c1 = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)

    x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c2 = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c2)
    x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c3 = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(c3)
    x = concatenate([c2, x])
    x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = concatenate([c1, x])

    x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)

    output = TimeDistributed(Convolution2D(1, 3, 3, border_mode='same', activation='sigmoid', name='output'))(x)
    model = Model(inputs=[input_img], outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def LSTMNET2(input_shape):
    c = 16
    input_img = Input((None,256, 256, 1), name='input')
    x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(input_img)
    c1 = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)
    x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c2 = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c2)
    x = ConvLSTM2D(nb_filter=2*c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c3 = ConvLSTM2D(nb_filter=2*c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    
    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c3)
    x = ConvLSTM2D(nb_filter=3*c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c4 = ConvLSTM2D(nb_filter=3*c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    
    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c4)
    x = ConvLSTM2D(nb_filter=4*c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c5 = ConvLSTM2D(nb_filter=4*c, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    

    x = TimeDistributed(UpSampling2D((2, 2)))(c5)
    x = concatenate([c4, x])
    x = TimeDistributed(Convolution2D(3*c, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(Convolution2D(3*c, 3, 3, border_mode='same'))(x)
    
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = concatenate([c3, x])
    x = TimeDistributed(Convolution2D(2*c, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(Convolution2D(2*c, 3, 3, border_mode='same'))(x)

    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = concatenate([c2, x])
    x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(Convolution2D(c, 3, 3, border_mode='same'))(x)
    
    x = TimeDistributed(UpSampling2D((2, 2)))(x)
    x = concatenate([c1, x])

    x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)

    output = TimeDistributed(Convolution2D(1, 3, 3, border_mode='same', activation='sigmoid', name='output'))(x)
    model = Model(inputs=[input_img], outputs=[output])
    # opti = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def LSTMNET3(input_shape, filters=16):
    input_img = Input(input_shape, name='input')
    x = ConvLSTM2D(nb_filter=filters, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(input_img)
    c1 = ConvLSTM2D(nb_filter=filters, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)
    x = ConvLSTM2D(nb_filter=filters, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c2 = ConvLSTM2D(nb_filter=filters, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c2)
    x = ConvLSTM2D(nb_filter=2*filters, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c3 = ConvLSTM2D(nb_filter=2*filters, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    
    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c3)
    x = ConvLSTM2D(nb_filter=3*filters, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c4 = ConvLSTM2D(nb_filter=3*filters, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    
    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c4)
    x = ConvLSTM2D(nb_filter=4*filters, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    c5 = ConvLSTM2D(nb_filter=4*filters, nb_row=3, nb_col=3, border_mode='same', return_sequences=True)(x)
    

    x = TimeDistributed(Conv2DTranspose(3*filters, (2, 2), strides=(2, 2), padding='same'))(c5)
    x = concatenate([c4, x])
    x = TimeDistributed(Convolution2D(3*filters, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(Convolution2D(3*filters, 3, 3, border_mode='same'))(x)
    
    x = TimeDistributed(Conv2DTranspose(2*filters, (2, 2), strides=(2, 2), padding='same'))(x)
    x = concatenate([c3, x])
    x = TimeDistributed(Convolution2D(2*filters, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(Convolution2D(2*filters, 3, 3, border_mode='same'))(x)

    x = TimeDistributed(Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same'))(x)
    x = concatenate([c2, x])
    x = TimeDistributed(Convolution2D(filters, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(Convolution2D(filters, 3, 3, border_mode='same'))(x)
    
    x = TimeDistributed(Conv2DTranspose(3, (2, 2), strides=(2, 2), padding='same'))(x)
    x = concatenate([c1, x])

    x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)
    x = TimeDistributed(Convolution2D(3, 3, 3, border_mode='same'))(x)

    output = TimeDistributed(Convolution2D(1, 3, 3, border_mode='same', activation='sigmoid', name='output'))(x)
    model = Model(inputs=[input_img], outputs=[output])
    # opti = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer='adadelta')
    return model


def Unet2D(input_size = (256,256,1),filters=64):
    inputs = Input(input_size)
    conv1 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(filters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(filters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(filters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(filters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(filters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(filters*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(filters*16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(filters*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(filters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(filters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(filters*4, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(filters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(filters*4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(filters*2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(filters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(filters*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)


    return model