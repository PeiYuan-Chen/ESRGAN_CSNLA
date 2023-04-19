import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, UpSampling2D, Flatten, Dense, add, Lambda
from tensorflow.keras.models import Model
from utils.conv2D_args import k3n64s1, k3n3s1
from models.backbone.RRDB import residual_in_residual_channel_attention_dense_block
from train_utils.sn import SpectralNormalization
from models.attention import in_scale_non_local_attention_residual_block


def generator(kernel_initializer=tf.keras.initializers.GlorotNormal()):
    inputs = Input(shape=(None, None, 3))
    # pre-process
    x = tf.keras.layers.Rescaling(scale=1.0 / 255)(inputs)
    # shallow extraction
    x = Conv2D(kernel_initializer=kernel_initializer, **k3n64s1)(x)
    x = in_scale_non_local_attention_residual_block(
        input_tensor=x, channel_reduction=2, softmax_factor=6, kernel_initializer=kernel_initializer)

    # trunk
    lsc = x
    for _ in range(6):
        x = residual_in_residual_channel_attention_dense_block(
            x, kernel_initializer=kernel_initializer)
    x = in_scale_non_local_attention_residual_block(
        input_tensor=x, channel_reduction=2, softmax_factor=6, kernel_initializer=kernel_initializer)

    for _ in range(6):
        x = residual_in_residual_channel_attention_dense_block(
            x, kernel_initializer=kernel_initializer)
    x = in_scale_non_local_attention_residual_block(
        input_tensor=x, channel_reduction=2, softmax_factor=6, kernel_initializer=kernel_initializer)

    for _ in range(6):
        x = residual_in_residual_channel_attention_dense_block(
            x, kernel_initializer=kernel_initializer)
    x = in_scale_non_local_attention_residual_block(
        input_tensor=x, channel_reduction=2, softmax_factor=6, kernel_initializer=kernel_initializer)

    for _ in range(5):
        x = residual_in_residual_channel_attention_dense_block(
            x, kernel_initializer=kernel_initializer)

    x = Conv2D(kernel_initializer=kernel_initializer, **k3n64s1)(x)
    x = add([x, lsc])

    x = in_scale_non_local_attention_residual_block(
        input_tensor=x, channel_reduction=2, softmax_factor=6, kernel_initializer=kernel_initializer)
    # upsample nearest
    x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    x = Conv2D(kernel_initializer=kernel_initializer, **k3n64s1)(x)
    x = LeakyReLU(alpha=0.2)(x)

    # reconstruct
    x = Conv2D(kernel_initializer=kernel_initializer, **k3n64s1)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(kernel_initializer=kernel_initializer, **k3n3s1)(x)

    # post-process
    outputs = tf.keras.layers.Rescaling(scale=255)(x)
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model


def generator_x4(kernel_initializer=tf.keras.initializers.GlorotNormal()):
    # inputs = Input(shape=(input_height, input_width, 3))
    inputs = Input(shape=(None, None, 3))
    # pre-process
    x = tf.keras.layers.Rescaling(scale=1.0 / 255)(inputs)

    # shallow extraction
    x = Conv2D(kernel_initializer=kernel_initializer, **k3n64s1)(x)

    x = in_scale_non_local_attention_residual_block(
        input_tensor=x, channel_reduction=2, softmax_factor=6, kernel_initializer=kernel_initializer)

    # trunk
    lsc = x
    for _ in range(6):
        x = residual_in_residual_channel_attention_dense_block(
            x, kernel_initializer=kernel_initializer)
    x = in_scale_non_local_attention_residual_block(
        input_tensor=x, channel_reduction=2, softmax_factor=6, kernel_initializer=kernel_initializer)

    for _ in range(6):
        x = residual_in_residual_channel_attention_dense_block(
            x, kernel_initializer=kernel_initializer)
    x = in_scale_non_local_attention_residual_block(
        input_tensor=x, channel_reduction=2, softmax_factor=6, kernel_initializer=kernel_initializer)

    for _ in range(6):
        x = residual_in_residual_channel_attention_dense_block(
            x, kernel_initializer=kernel_initializer)
    x = in_scale_non_local_attention_residual_block(
        input_tensor=x, channel_reduction=2, softmax_factor=6, kernel_initializer=kernel_initializer)

    for _ in range(5):
        x = residual_in_residual_channel_attention_dense_block(
            x, kernel_initializer=kernel_initializer)

    x = Conv2D(kernel_initializer=kernel_initializer, **k3n64s1)(x)
    x = add([x, lsc])

    x = in_scale_non_local_attention_residual_block(
        input_tensor=x, channel_reduction=2, softmax_factor=6, kernel_initializer=kernel_initializer)
    # upsample nearest
    x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    x = Conv2D(kernel_initializer=kernel_initializer, **k3n64s1)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = UpSampling2D(size=(2, 2), interpolation='nearest',
                     name='additional_start_layer')(x)
    x = Conv2D(kernel_initializer=kernel_initializer, **k3n64s1)(x)
    x = LeakyReLU(alpha=0.2, name='additional_end_layer')(x)

    # reconstruct
    x = Conv2D(kernel_initializer=kernel_initializer, **k3n64s1)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(kernel_initializer=kernel_initializer, **k3n3s1)(x)

    # post-process
    outputs = tf.keras.layers.Rescaling(scale=255)(x)
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model


def discriminator_model(filter_num=64):
    inputs = Input(shape=(128, 128, 3))
    # spatial_size=(128,128)
    inputs = tf.keras.layers.Rescaling(scale=1.0 / 255)(inputs)
    # shallow extraction
    x = Conv2D(filters=filter_num,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same', )(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    # downsample
    x = Conv2D(filters=filter_num,
               kernel_size=(4, 4),
               strides=(2, 2),
               padding='same',
               use_bias=False)(x)  # output size /2
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=filter_num * 2,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               use_bias=False)(x)  # output feature map * 2
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=filter_num * 2,
               kernel_size=(4, 4),
               strides=(2, 2),
               padding='same',
               use_bias=False)(x)  # output size /4
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=filter_num * 4,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               use_bias=False)(x)  # output feature map * 4
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=filter_num * 4,
               kernel_size=(4, 4),
               strides=(2, 2),
               padding='same',
               use_bias=False)(x)  # output size /8
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=filter_num * 8,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               use_bias=False)(x)  # output feature map * 8
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=filter_num * 8,
               kernel_size=(4, 4),
               strides=(2, 2),
               padding='same',
               use_bias=False)(x)  # output size /16
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=filter_num * 8,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same',
               use_bias=False)(x)  # output feature map * 16
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(filters=filter_num * 8,
               kernel_size=(4, 4),
               strides=(2, 2),
               padding='same',
               use_bias=False)(x)  # output size /32
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # spatial size=(4,4)
    x = Flatten()(x)
    x = Dense(units=100)(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Dense(units=1)(x)
    # model
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model


def discriminator_model_sn(filter_num=64):
    inputs = Input(shape=(128, 128, 3))
    # spatial_size=(128,128)
    inputs = tf.keras.layers.Rescaling(scale=1.0 / 255)(inputs)
    # shallow extraction
    x = Conv2D(filters=filter_num,
               kernel_size=(3, 3),
               strides=(1, 1),
               padding='same', )(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    # downsample
    x = SpectralNormalization(Conv2D(filters=filter_num,
                                     kernel_size=(4, 4),
                                     strides=(2, 2),
                                     padding='same',
                                     use_bias=False))(x)  # output size /2
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = SpectralNormalization(Conv2D(filters=filter_num * 2,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     use_bias=False))(x)  # output feature map * 2
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = SpectralNormalization(Conv2D(filters=filter_num * 2,
                                     kernel_size=(4, 4),
                                     strides=(2, 2),
                                     padding='same',
                                     use_bias=False))(x)  # output size /4
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = SpectralNormalization(Conv2D(filters=filter_num * 4,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     use_bias=False))(x)  # output feature map * 4
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = SpectralNormalization(Conv2D(filters=filter_num * 4,
                                     kernel_size=(4, 4),
                                     strides=(2, 2),
                                     padding='same',
                                     use_bias=False))(x)  # output size /8
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = SpectralNormalization(Conv2D(filters=filter_num * 8,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     use_bias=False))(x)  # output feature map * 8
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = SpectralNormalization(Conv2D(filters=filter_num * 8,
                                     kernel_size=(4, 4),
                                     strides=(2, 2),
                                     padding='same',
                                     use_bias=False))(x)  # output size /16
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = SpectralNormalization(Conv2D(filters=filter_num * 8,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='same',
                                     use_bias=False))(x)  # output feature map * 16
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = SpectralNormalization(Conv2D(filters=filter_num * 8,
                                     kernel_size=(4, 4),
                                     strides=(2, 2),
                                     padding='same',
                                     use_bias=False))(x)  # output size /32
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # spatial size=(4,4)
    x = Flatten()(x)
    x = Dense(units=100)(x)
    x = LeakyReLU(alpha=0.2)(x)
    outputs = Dense(units=1)(x)
    # model
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model
