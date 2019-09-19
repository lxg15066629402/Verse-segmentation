# -*-encoding: utf-8 -*-

"""
@version: v2.0
@author: Lxg
@time: 2019/08/15
@email: lvxiaogang0428@163.com
@function: network method
"""
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, add, Conv3DTranspose, AveragePooling3D, ZeroPadding3D, Activation, Dropout, GaussianNoise
from keras.optimizers import Adam
from keras import backend as K
from loss import *
import numpy as np
from keras.layers.normalization import BatchNormalization

K.set_image_data_format('channels_last')
project_name = '3DUNet'


def get_unet(input_size=(128, 96, 64, 1)):
    # inputs = Input((img_depth, img_rows, img_cols, 1)
    inputs = Input(input_size)

    input = BatchNormalization()(inputs)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    Save = Conv3D(64, (1, 1, 1), activation='relu', padding='same')(conv1)
    conv1 = add([Save, conv1])
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    pool1 = BatchNormalization()(pool1)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    Save = Conv3D(128, (1, 1, 1), activation='relu', padding='same')(conv2)
    conv2 = add([Save, conv2])
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    pool2 = BatchNormalization()(pool2)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    Save = Conv3D(256, (1, 1, 1), activation='relu', padding='same')(conv3)
    conv3 = add([Save, conv3])
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    pool3 = BatchNormalization()(pool3)
    conv4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv4)
    Save = Conv3D(512, (1, 1, 1), activation='relu', padding='same')(conv4)
    conv4 = add([Save, conv4])
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    #

    pool4 = BatchNormalization()(pool4)
    conv5 = Conv3D(1024, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv3D(1024, (3, 3, 3), activation='relu', padding='same')(conv5)
    conv5 = Dropout(0.5)(conv5)

    up6 = concatenate([Conv3DTranspose(512, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv4], axis=4)
    conv6 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv6)
    Save = Conv3D(512, (1, 1, 1), activation='relu', padding='same')(conv6)
    conv6 = add([Save, conv6])

    up7 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv3], axis=4)
    conv7 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv7)
    Save = Conv3D(256, (1, 1, 1), activation='relu', padding='same')(conv7)
    conv7 = add([Save, conv7])

    up8 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv2], axis=4)
    conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv8)
    Save = Conv3D(128, (1, 1, 1), activation='relu', padding='same')(conv8)
    conv8 = add([Save, conv8])

    up9 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), conv1], axis=4)
    conv9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv9)
    Save = Conv3D(64, (1, 1, 1), activation='relu', padding='same')(conv9)
    conv9 = add([Save, conv9])

    conv10 = Conv3D(26, (1, 1, 1), activation='softmax', padding='same')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.summary()  # print model way

    model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss=dice_loss,
                  metrics=['acc', dice_score_metric])  # dice_coef  binary_crossentropy

    return model


# 1
# def get_unet(input_size=(96, 96, 96, 1)):
#     inputs = Input(input_size)
#     input = GaussianNoise(0.01)(inputs)
#     conv1 = Conv3D(64, (3, 3, 3), activation='relu',  padding='same')(input)
#     conv1 = Conv3D(64, (3, 3, 3), activation='relu',  padding='same')(conv1)
#     pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
#
#     conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
#     conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
#
#     conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
#     conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
#     pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
#
#     conv4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool3)
#     conv4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv4)
#     pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
#
#     conv5 = Conv3D(1024, (3, 3, 3), activation='relu', padding='same')(pool4)
#     conv5 = Conv3D(1024, (3, 3, 3), activation='relu', padding='same')(conv5)
#     conv5 = Dropout(0.5)(conv5)  # 0.2
#
#     up6 = concatenate([Conv3DTranspose(512, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv4], axis=4)
#     conv6 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(up6)
#     conv6 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv6)
#
#     up7 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv3], axis=4)
#     conv7 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up7)
#     conv7 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv7)
#
#     up8 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv2], axis=4)
#     conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up8)
#     conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv8)
#
#     up9 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), conv1], axis=4)
#     conv9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up9)
#     conv9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv9)
#
#     conv10 = Conv3D(25, (1, 1, 1), activation='softmax', padding='same')(conv9)
#
#     model = Model(inputs=[inputs], outputs=[conv10])
#
#     model.summary()  # print model way
#
#     # model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss=dice_loss, metrics=['accuracy', dice])  # dice_coef  binary_crossentropy
#     model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss=dice_loss, metrics=['accuracy', dice])  # dice_coef  binary_crossentropy
#
#     return model

# def get_unet(input_size=(96, 96, 96, 1)):
#     inputs = Input(input_size)
#     input = GaussianNoise(0.01)(inputs)
#     conv1 = Conv3D(64, (3, 3, 3), activation='relu',  padding='same')(input)
#     conv1 = Conv3D(64, (3, 3, 3), activation='relu',  padding='same')(conv1)
#     pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
#
#     conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
#     conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
#
#     conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
#     conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
#     pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
#
#     conv4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool3)
#     conv4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv4)
#     conv4 = Dropout(0.5)(conv4)
#
#     up7 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4), conv3], axis=4)
#     conv7 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up7)
#     conv7 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv7)
#
#     up8 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv2], axis=4)
#     conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up8)
#     conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv8)
#
#     up9 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), conv1], axis=4)
#     conv9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up9)
#     conv9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv9)
#
#     conv10 = Conv3D(25, (1, 1, 1), activation='softmax', padding='same')(conv9)
#
#     model = Model(inputs=[inputs], outputs=[conv10])
#
#     model.summary()  # print model way
#
#     # model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss=dice_loss, metrics=['accuracy', dice])  # dice_coef  binary_crossentropy
#     model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss=dice_loss, metrics=['accuracy', dice])  # dice_coef  binary_crossentropy
#
#     return model
