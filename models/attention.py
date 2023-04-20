import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Reshape, Softmax, add, PReLU, Layer, GlobalAveragePooling2D, multiply


class CrossScaleNonLocalAttention(Layer):
    def __init__(self, channel_reduction=2, scale=4, patch_size=3, softmax_factor=10, kernel_initializer=tf.keras.initializers.GlorotNormal(), **kwargs):
        super(CrossScaleNonLocalAttention, self).__init__(**kwargs)
        self.channel_reduction = channel_reduction
        self.scale = scale
        self.patch_size = patch_size
        self.softmax_factor = softmax_factor
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        channels = input_shape[-1]
        inter_channels = channels // self.channel_reduction

        self.theta = Conv2D(filters=inter_channels,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer=self.kernel_initializer,)
        self.theta_PRelu = PReLU(shared_axes=[1, 2])

        self.phi = Conv2D(filters=inter_channels,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding='same',
                          kernel_initializer=self.kernel_initializer,)
        self.phi_PRelu = PReLU(shared_axes=[1, 2])

        self.g = Conv2D(filters=channels,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=self.kernel_initializer,)
        self.g_PRelu = PReLU(shared_axes=[1, 2])
        return super().build(input_shape)

    # @tf.function
    def call(self, inputs, *args, **kwargs):
        input_shape = tf.shape(inputs)
        batch_size, height, width, channels = input_shape[
            0], input_shape[1], input_shape[2], input_shape[3]
        inter_height, inter_width, inter_channels = height // self.scale, width // self.scale, channels // self.channel_reduction
        # theta = self.theta_PRelu(self.theta(inputs))  # (b,h,w,c/2)
        theta = self.theta_PRelu(self.theta(inputs))  # (b,h,w,c/2)
        phi = tf.image.resize(inputs, size=(inter_height, inter_width),
                              method=tf.image.ResizeMethod.BILINEAR)  # (b,h/s,w/s,c)
        # phi = self.phi_PRelu(self.phi(phi))  # (b,h/s,w/s,c/2)
        phi = self.phi_PRelu(self.phi(phi))  # (b,h/s,w/s,c/2)
        phi_patch = tf.image.extract_patches(images=phi,
                                             sizes=(1, self.patch_size,
                                                    self.patch_size, 1),
                                             strides=(1, 1, 1, 1),
                                             rates=(1, 1, 1, 1),
                                             padding='SAME')  # (b,h/s,w/s,p*p*c/2)
        phi_patch = tf.reshape(tensor=phi_patch,
                               shape=(-1, inter_height*inter_width, self.patch_size, self.patch_size, inter_channels))
        # (b,N,p,p,c/2) N = hw/(s*s)
        # g = self.g_PRelu(self.g(inputs))  # (b,h,w,c)
        g = self.g_PRelu(self.g(inputs))  # (b,h,w,c)
        g_patch = tf.image.extract_patches(images=g,
                                           sizes=(1, self.scale*self.patch_size,
                                                  self.scale*self.patch_size, 1),
                                           strides=(1, self.scale,
                                                    self.scale, 1),
                                           rates=(1, 1, 1, 1),
                                           padding='SAME')  # (b,h/s,w/s,s*p*s*p*c)
        g_patch = tf.reshape(tensor=g_patch,
                             shape=(-1, inter_height*inter_width, self.scale*self.patch_size, self.scale*self.patch_size, channels))
        # (b,N,s*p,s*p,c) N = hw/(s*s)
        # phi_patch = tf.unstack(phi_patch, axis=0)
        # g_patch = tf.unstack(g_patch, axis=0)

        def process_patches(args):
            theta_i, phi_patch_i, g_patch_i = args
            theta_i = tf.expand_dims(theta_i, axis=0)  # (1,h,w,c/2)

            max_phi_patch_i = tf.sqrt(tf.reduce_sum(
                tf.square(phi_patch_i), axis=[1, 2, 3], keepdims=True))
            max_phi_patch_i = tf.maximum(max_phi_patch_i, 1e-6)
            phi_patch_i = phi_patch_i / max_phi_patch_i

            phi_patch_i = tf.transpose(phi_patch_i,
                                       perm=(1, 2, 3, 0))  # (p,p,c/2,N)

            y_i = tf.nn.conv2d(input=theta_i,
                               filters=phi_patch_i,
                               strides=(1, 1),
                               padding='SAME',
                               data_format='NHWC')  # (1,h,w,N)
            y_i_softmax = tf.nn.softmax(
                y_i*self.softmax_factor, axis=-1)  # feature map

            # g_patch_i (s*p,s*p,c,N)
            g_patch_i = tf.transpose(g_patch_i, perm=[1, 2, 3, 0])
            y_i = tf.nn.conv2d_transpose(input=y_i_softmax,
                                         filters=g_patch_i,
                                         output_shape=(
                                             1, self.scale*height, self.scale*width, channels),
                                         strides=self.scale,
                                         padding='SAME',
                                         data_format='NHWC')  # (1,s*h,s*w,c)
            y_i = y_i / 6
            return y_i

        y = tf.map_fn(process_patches, (theta, phi_patch,
                      g_patch), dtype=tf.float32)
        y = tf.squeeze(y, axis=1)
        output_shape = (batch_size, self.scale * height,
                        self.scale * width, channels)
        y = tf.reshape(y, output_shape)
        return y

        # y = []
        # for theta_i, phi_patch_i, g_patch_i in zip(theta, phi_patch, g_patch):
        #     theta_i = tf.expand_dims(theta_i, axis=0)  # (1,h,w,c/2
        #     # theta_i (1,h,w,c/2) split
        #     # phi_patch_i (N,p,p,c/2) unstack
        #     # g_patch_i (N,s*p,s*p,c) unstack
        #     # normalize phi_patch_i
        #     max_phi_patch_i = tf.sqrt(tf.reduce_sum(
        #         tf.square(phi_patch_i), axis=[1, 2, 3], keepdims=True))
        #     max_phi_patch_i = tf.maximum(max_phi_patch_i, 1e-6)
        #     phi_patch_i = phi_patch_i / max_phi_patch_i

        #     phi_patch_i = tf.transpose(phi_patch_i,
        #                                perm=(1, 2, 3, 0))  # (p,p,c/2,N)

        #     y_i = tf.nn.conv2d(input=theta_i,
        #                        filters=phi_patch_i,
        #                        strides=(1, 1),
        #                        padding='same',
        #                        data_format='NHWC')  # (1,h,w,N)
        #     y_i_softmax = tf.nn.softmax(
        #         y_i*self.softmax_factor, axis=-1)  # feature map

        #     # g_patch_i (s*p,s*p,c,N)
        #     g_patch_i = tf.transpose(g_patch_i, perm=[1, 2, 3, 0])
        #     y_i = tf.nn.conv2d_transpose(input=y_i_softmax,
        #                                  filters=g_patch_i,
        #                                  output_shape=(
        #                                      1, self.scale*height, self.scale*width, channels),
        #                                  strides=self.scale,
        #                                  padding='SAME',
        #                                  data_format='NHWC')  # (1,s*h,s*w,c)
        #     y_i = y_i / 6
        #     y.append(y_i)
        # y = tf.concat(y, axis=0)
        # return y


class InsclaeNonLocalAttention(Layer):
    def __init__(self, channel_reduction=2,  softmax_factor=6, kernel_initializer=tf.keras.initializers.GlorotNormal(), **kwargs):
        super(InsclaeNonLocalAttention, self).__init__(**kwargs)
        self.channel_reduction = channel_reduction
        self.softmax_factor = softmax_factor
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        channels = input_shape[-1]
        inter_channels = channels // self.channel_reduction

        self.theta = Conv2D(filters=inter_channels,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer=self.kernel_initializer,)

        self.phi = Conv2D(filters=inter_channels,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding='same',
                          kernel_initializer=self.kernel_initializer,)
        self.g = Conv2D(filters=inter_channels,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=self.kernel_initializer,)
        self.y = Conv2D(filters=channels,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=self.kernel_initializer,)

        return super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        dynamic_shape = tf.shape(inputs)
        _, height, width, channels = dynamic_shape[
            0], dynamic_shape[1], dynamic_shape[2], dynamic_shape[3]

        inter_channels = channels // self.channel_reduction
        theta = self.theta(inputs)  # (b,h,w,c/2)
        phi = self.phi(inputs)  # (b,h,w,c/2)
        g = self.g(inputs)  # (b,h,w,c/2)

        theta_flat = tf.reshape(theta, shape=(
            -1, height*width, inter_channels))  # (b,h*w,c/2)
        phi_flat = tf.reshape(phi, shape=(
            -1, height*width, inter_channels))  # (b,h*w,c/2)
        g_flat = tf.reshape(
            g, shape=(-1, height*width, inter_channels))  # (b,h*w,c/2)
        # theta_flat = self.reshape_flat(theta)  # (b,h*w,c/2)
        # phi_flat = self.reshape_flat(phi)  # (b,h*w,c/2)
        # g_flat = self.reshape_flat(g)  # (b,h*w,c/2)

        attention_map = tf.matmul(
            theta_flat, phi_flat, transpose_b=True)  # (b,h*w,h*w)
        attention_map_softmax = tf.nn.softmax(
            attention_map*self.softmax_factor, axis=-1)  # feature map

        y = tf.matmul(attention_map_softmax, g_flat)  # (b,h*w,c/2)
        # y = Reshape(target_shape=(height, width, inter_channels))(y)  # (b,h,w,c/2)
        y = tf.reshape(y, shape=(-1, height, width, inter_channels))
        y = self.y(y)  # (b,h,w,c)
        return y


def in_scale_non_local_attention_residual_block(input_tensor, channel_reduction=2, softmax_factor=6, kernel_initializer=tf.keras.initializers.GlorotNormal()):
    x = InsclaeNonLocalAttention(channel_reduction=channel_reduction,
                                 softmax_factor=softmax_factor,
                                 kernel_initializer=kernel_initializer,)(input_tensor)
    return add([input_tensor, x])


class ChannelAttention(Layer):
    def __init__(self, reduction=16, kernel_initializer=tf.keras.initializers.GlorotNormal(), **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction = reduction
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.conv1 = Conv2D(filters=input_shape[-1]//self.reduction,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            kernel_initializer=self.kernel_initializer,)
        self.conv2 = Conv2D(filters=input_shape[-1],
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='same',
                            activation='sigmoid',
                            kernel_initializer=self.kernel_initializer,)

        self.reshape = Reshape(target_shape=(1, 1, input_shape[-1]))
        return super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = self.avg_pool(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.reshape(x)
        return multiply([inputs, x])
