# -*- coding: utf-8 -*-
from keras.layers import Activation, Conv2D, GlobalAveragePooling2D, Dense, Reshape, multiply, Multiply
import keras.backend as K
import tensorflow as tf
from keras.layers import Layer

# Dual Attention Network for Scene Segmentation, CVPR2019
#https://github.com/niecongchong/DANet-keras
class PAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(PAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        b = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input)
        c = Conv2D(filters // 8, 1, use_bias=False, kernel_initializer='he_normal')(input)
        d = Conv2D(filters, 1, use_bias=False, kernel_initializer='he_normal')(input)

        vec_b = K.reshape(b, (-1, h * w, filters // 8))
        vec_cT = tf.transpose(K.reshape(c, (-1, h * w, filters // 8)), (0, 2, 1))
        bcT = K.batch_dot(vec_b, vec_cT)
        softmax_bcT = Activation('softmax')(bcT)
        vec_d = K.reshape(d, (-1, h * w, filters))
        bcTd = K.batch_dot(softmax_bcT, vec_d)
        bcTd = K.reshape(bcTd, (-1, h, w, filters))

        out = self.gamma*bcTd + input
        return out


class CAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(CAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        vec_a = K.reshape(input, (-1, h * w, filters))
        vec_aT = tf.transpose(vec_a, (0, 2, 1))
        aTa = K.batch_dot(vec_aT, vec_a)
        softmax_aTa = Activation('softmax')(aTa)
        aaTa = K.batch_dot(vec_a, softmax_aTa)
        aaTa = K.reshape(aaTa, (-1, h, w, filters))

        out = self.gamma*aaTa + input
        return out

# squeeze-excitation layer (channel-wise attention layer)
# Self-Critical Attention Learning for Person Re-Identification, ICCV2019
# BAM: Bottleneck Attention Module, BMVC2018 
# Squeeze-and-Excitation Netoworks, CVPR2018
class SE(Layer):
    def __init__(self, **kwargs):
        super(SE, self).__init__(**kwargs)


    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        # first apply GAP
        squeeze = GlobalAveragePooling2D()(input)
        # then squeeze-excite
        excitation = Dense(filters //4, kernel_initializer='he_normal')(squeeze) # default is 4
        excitation = Activation('relu')(excitation)
        excitation = Dense(filters, kernel_initializer='he_normal')(excitation)
        excitation = Activation('sigmoid')(excitation)
        excitation = Reshape((1,1,filters))(excitation)

        # perform multiplication
        scale = multiply([input,excitation])
        return scale

# self-attention layer
# Non-local Neural Networks, CVPR2018
# Self-Attention Generative Adversarial Networks. PMLR 2019
# https://github.com/kiyohiro8/SelfAttentionGAN
class SelfAttention(Layer):

    def __init__(self, channels, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = channels
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels


    def build(self, input_shape):
        # shape=(self.kernel_size+(num_channels,self.filters))
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='he_normal',
                                        name='kernel_f',
                                        trainable=True) # (1,1,2048,256)
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='he_normal',
                                        name='kernel_g',
                                        trainable=True)
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='he_normal',
                                        name='kernel_h',
                                        trainable=True) # (1,1,2048,2048)

        #super(SelfAttention, self).build(input_shape)

        self.built = True

    def call(self, x):
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[3]])

        f = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        h = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]

        s = K.batch_dot(hw_flatten(g), K.permute_dimensions(hw_flatten(f), (0, 2, 1)))  # # [bs, N, N]

        beta = K.softmax(s, axis=-1)  # attention map

        o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]

        o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


# implements the SA and CA layers as identified in the following paper
# self-critical attention learning for person re-identification, ICCV2019
class SA(Layer):
    def __init__(self, **kwargs):
        super(SA, self).__init__(**kwargs)


    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        input_shape = input.get_shape().as_list()
        _, h, w, filters = input_shape

        # Compute the average across the channel domains of the feature maps
        squeeze = Lambda(lambda x: K.mean(x, axis=2))(input)
        # Flatten the mean map
        squeeze = Flatten()(squeeze)
        # then squeeze-excite
        excitation = Dense(filters //4, kernel_initializer='he_normal')(squeeze) # default is 4
        excitation = Activation('relu')(excitation)
        excitation = Dense(filters, kernel_initializer='he_normal')(excitation)
        excitation = Activation('sigmoid')(excitation)
        excitation = Reshape((h,w,filters))(excitation)

        # perform multiplication
        scale = Multiply([input,excitation])
        return scale


