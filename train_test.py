# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import SimpleITK as sitk
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.callbacks import TensorBoard
from keras import backend as K
from model import get_unet
K.set_image_data_format('channels_last')
project_name = '3DUNet'


def metr_dice(s, g):
    # dice score of two 3D volumes
    num = np.sum(np.multiply(s, g))
    denom = s.sum() + g.sum()
    if denom == 0:
        return 1
    else:
        return 2.0 * num / denom


# train
def train(train_data, train_label):
    print('---'*30)
    print('Loading and preprocessing train data...')
    print('---'*30)

    print('---'*30)
    print('Creating and compiling model...')
    print('---'*30)
    model = get_unet((128, 96, 64, 1))

    model_checkpoint = ModelCheckpoint(filepath='model/ResUnet.hdf5', save_best_only=True, verbose=1, monitor='dice_score_metric', mode='max')

    log_dir = 'model/logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    print('---'*30)
    print('Fitting model...')
    print('---'*30)        # 25
    # model.fit(train_data, train_label, batch_size=1, epochs=30, verbose=1, shuffle=True, validation_split=0.10, callbacks=[model_checkpoint, TensorBoard(log_dir="model/logs")])
    # model.fit(train_data, train_label, batch_size=1, epochs=600, verbose=1, shuffle=True, callbacks=[model_checkpoint, TensorBoard(log_dir="model/logs")])

    model.fit(train_data, train_label, batch_size=1, epochs=1500
              , verbose=1, shuffle=True, validation_split=0.10, callbacks=[model_checkpoint, TensorBoard(log_dir="model/logs")])

    print('---'*30)
    print('Training finished')
    print('---'*30)


if __name__ == '__main__':

    print("loading dataset done\n")
    # load npy data
    data = np.load("x_training.npy").astype(np.float32)
    labels = np.load("y_training.npy").astype(np.uint8)

    # shuffle = list(zip(data, labels))
    # np.random.shuffle(shuffle)

    # 5-cross validation
    # 01
    # train_data, train_labels = data[: int(len(shuffle) * 0.8)], labels[:int(len(shuffle) * 0.8)]
    # test_data, test_label = data[int(len(shuffle) * 0.8):], labels[int(len(shuffle) * 0.8):]

    # 02
    # train_data, train_labels = data[int(len(shuffle) * 0.2):], labels[:int(len(shuffle) * 0.8)]
    # test_data, test_label = data[:int(len(shuffle) * 0.2)], labels[:int(len(shuffle) * 0.2)]

    # 03
    # train_data1, train_labels1 = data[: int(len(shuffle) * 0.5)], labels[: int(len(shuffle) * 0.5)]
    # train_data2, train_labels2 = data[int(len(shuffle) * 0.7):], labels[int(len(shuffle) * 0.7):]
    # train_data = np.concatenate((train_data1, train_data2), axis=0)
    # train_labels = np.concatenate((train_labels1, train_labels2), axis=0)
    # test_data, test_label = data[int(len(shuffle) * 0.5):int(len(shuffle) * 0.7)], labels[int(len(shuffle) * 0.5):int(len(shuffle) * 0.7)]

    # 04
    # train_data1, train_labels1 = data[: int(len(shuffle) * 0.7)], labels[: int(len(shuffle) * 0.7)]
    # train_data2, train_labels2 = data[int(len(shuffle) * 0.9):], labels[int(len(shuffle) * 0.9):]
    # train_data = np.concatenate((train_data1, train_data2), axis=0)
    # train_labels = np.concatenate((train_labels1, train_labels2), axis=0)
    # test_data, test_label = data[int(len(shuffle) * 0.7):int(len(shuffle) * 0.9)], labels[int(len(shuffle) * 0.7):int(len(shuffle) * 0.9)]

    # 05
    # train_data1, train_labels1 = data[: int(len(shuffle) * 0.3)], labels[: int(len(shuffle) * 0.3)]
    # train_data2, train_labels2 = data[int(len(shuffle) * 0.5):], labels[int(len(shuffle) * 0.5):]
    # train_data = np.concatenate((train_data1, train_data2), axis=0)
    # train_labels = np.concatenate((train_labels1, train_labels2), axis=0)
    # test_data, test_label = data[int(len(shuffle) * 0.3):int(len(shuffle) * 0.5)], labels[int(len(shuffle) * 0.3):int(
    #     len(shuffle) * 0.5)]

    # all train data set
    train_data = data
    train_labels = labels

    train(train_data, train_labels)
    print("train finished")
