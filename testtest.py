import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model


class Shape_Layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Shape_Layer, self).__init__(**kwargs)

    def call(self, inputs):
        s = tf.shape(inputs)
        height, width, channels = s[1], s[2], s[3]
        print(tf.shape(inputs))
        print(f"height: {height}, width: {width}, channels: {channels}")
        return inputs


inputs = Input(shape=(None, None, 3))
x = Conv2D(64, 3, padding='same')(inputs)
x = Shape_Layer()(x)
x = Conv2D(64, 3, padding='same')(x)
model = Model(inputs=inputs, outputs=x)

example_lr_img = tf.random.normal((1, 32, 32, 3))
output = model(example_lr_img, training=True)
