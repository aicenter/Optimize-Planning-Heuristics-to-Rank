import numpy as np
import math

from twodattention import AttentionAugmentation2D

import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Conv2D, Dense, Flatten, Input, concatenate, Lambda, Softmax, Activation, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

class AddPositionalEncoding(Layer):
    """
    Injects positional encoding signal described in section 3.5 of the original
    paper "Attention is all you need". Also a base class for more complex
    coordinate encoding described in "Universal Transformers".
    """

    def __init__(self,dim, **kwargs):
        self.dim = dim
        self.signal = None
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        return config

    def build(self, input_shape):
        _, height, width, channels = input_shape
        self.signal = posencode2d(
            self.dim, self.dim, channels)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.signal

def posencode2d(height,width,channels):
    """
    :param channels: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: height*width*channels position matrix
    """
    if channels % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(channels))
    pe = np.zeros((height, width, channels), dtype=K.floatx())
    channels = int(channels / 2) # 4
    div_term = K.exp(K.arange(0., channels, 2) *
                         -(math.log(10000.0) / channels))
    #print(div_term)
    pos_w = K.expand_dims(K.arange(0., width),1)
    pos_h = K.expand_dims(K.arange(0., height),1)

    pe[:, :, 0:channels:2] = K.repeat_elements(K.expand_dims(K.sin(pos_h * div_term),1),width, 1)
    pe[:, :, 1:channels:2] = K.repeat_elements(K.expand_dims(K.cos(pos_h * div_term),1),width, 1)
    pe[:, :, channels::2] = K.repeat_elements(K.expand_dims(K.sin(pos_w * div_term),0),height, 0)
    pe[:, :, channels + 1::2] =K.repeat_elements(K.expand_dims(K.cos(pos_w * div_term),0),height, 0)
    return pe

class LevinN:
    def __init__(self,dim):

        self.model        = self.create_model(dim)

    def create_model(self,dim):
        inputA = Input(shape=(None,None,5))
        inputB = Input(shape=(None,None,5))

        inp =  concatenate([inputA, inputB], axis=3)#

        b = Conv2D(64, (3, 3), padding = "same", activation='relu', name = 'conv1')(inp)
        b1 = concatenate([b, inp], axis=3)

        c = Conv2D(64, (3, 3), padding = "same", activation='relu', name = 'conv2')(b1)
        c1 = concatenate([c, inp], axis=3)

        d = Conv2D(64, (3, 3), padding = "same", activation='relu', name = 'conv3')(c1)
        d1 = concatenate([d, inp], axis=3)

        e = Conv2D(64, (3, 3), padding = "same", activation='relu', name = 'conv4')(d1)
        e1 = concatenate([e, inp], axis=3)

        f = Conv2D(64, (3, 3), padding = "same", activation='relu', name = 'conv5')(e1)
        f1 = concatenate([f, inp], axis=3)

        g = Conv2D(64, (3, 3), padding = "same", activation='relu', name = 'conv6')(f1)
        g1 = concatenate([g, inp], axis=3)

        h = Conv2D(64, (3, 3), padding = "same", activation='relu', name = 'conv7')(g1)
        h1 = concatenate([h, inp], axis=3)

        ########################################################################################
        i=Conv2D(180, (3, 3), padding = "same", activation='relu', name = 'conv8-1')(h1)
        att8 = AttentionAugmentation2D(60,60,2)(i)
        att8 = concatenate([att8, AddPositionalEncoding(dim)(i), inp], axis=3)


        j=Conv2D(180, (3, 3), padding = "same", activation='relu', name = 'conv9-1')(att8)
        att9 = AttentionAugmentation2D(60,60,2)(j)
        att9 = concatenate([att9, AddPositionalEncoding(dim)(j), inp], axis=3)

        k=Conv2D(180, (3, 3), padding = "same", activation='relu', name = 'conv10-1')(att9)
        att10 = AttentionAugmentation2D(60,60,2)(k)
        att10 = concatenate([att10, AddPositionalEncoding(dim)(k), inp], axis=3)

        l=Conv2D(180, (3, 3), padding = "same", activation='relu', name = 'conv11-1')(att10)
        att11 = AttentionAugmentation2D(60,60,2)(l)
        att11 = concatenate([att11, AddPositionalEncoding(dim)(l), inp], axis=3)

        f1=GlobalAveragePooling2D()(att11)
        d1 = Dense(256,  activation='relu', name = 'dense-1')(f1)
        op1 = Dense(8, activation='softmax', name = 'op-1')(d1)
        ###############################################################################
        p=Conv2D(180, (3, 3), padding = "same", activation='relu', name = 'conv8-2')(h1)
        att15 = AttentionAugmentation2D(60,60,2)(p)
        att15 = concatenate([att15, AddPositionalEncoding(dim)(p), inp], axis=3)

        q=Conv2D(180, (3, 3), padding = "same", activation='relu', name = 'conv9-2')(att15)
        att16 = AttentionAugmentation2D(60,60,2)(q)
        att16 = concatenate([att16, AddPositionalEncoding(dim)(q), inp], axis=3)

        r=Conv2D(180, (3, 3), padding = "same", activation='relu', name = 'conv10-2')(att16)
        att17 = AttentionAugmentation2D(60,60,2)(r)
        att17 = concatenate([att17, AddPositionalEncoding(dim)(r),inp], axis=3)


        s=Conv2D(180, (3, 3), padding = "same", activation='relu', name = 'conv11-2')(att17)
        att18 = AttentionAugmentation2D(60,60,2)(s)
        att18 = concatenate([att18, AddPositionalEncoding(dim)(s), inp], axis=3)

        f2=GlobalAveragePooling2D()(att18)
        d2 = Dense(256,  activation='relu', name = 'dense-2')(f2)
        op2 = Dense(1,  name = 'op-2')(d2)
        model = Model([inputA, inputB], [op1,op2])
        return model
