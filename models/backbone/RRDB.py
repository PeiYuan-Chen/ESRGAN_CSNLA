import tensorflow as tf
from tensorflow.keras.layers import add
from models.backbone.RDB import residual_dense_block
from models.attention import ChannelAttention


def residual_in_residual_dense_block(x, kernel_initializer=tf.keras.initializers.GlorotNormal()):
    identity = x
    for _ in range(3):
        x = residual_dense_block(x, kernel_initializer=kernel_initializer)
    x = x * 0.2
    return add([x, identity])


def residual_in_residual_channel_attention_dense_block(x, kernel_initializer=tf.keras.initializers.GlorotNormal()):
    identity = x
    for _ in range(3):
        x = residual_dense_block(x, kernel_initializer=kernel_initializer)
    x = ChannelAttention(
        reduction=16, kernel_initializer=kernel_initializer)(x)
    return add([x, identity])
