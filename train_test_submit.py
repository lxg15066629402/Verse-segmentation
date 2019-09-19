# -*- coding: utf-8  -*-

"""
@version: v2.0
@author: Lxg
@time: 2019/08/12
@email: lvxiaogang0428@163.com
@function: read .nii.gz data
"""

from __future__ import print_function
import os
from skimage.io import imsave
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from skimage import transform
np.random.seed(256)
import tensorflow as tf
tf.set_random_seed(256)
from model import get_unet
from keras import backend as K
from glob import glob

K.set_image_data_format('channels_last')
project_name = '3DUNet'


# metric
def metr_dice(s, g):
    # dice score of two 3D volumes
    num = np.sum(np.multiply(s, g))
    denom = s.sum() + g.sum()
    if denom == 0:
        return 1
    else:
        return 2.0 * num / denom


# read test data
def read_test(filename):
    data = glob(filename + '/*.nii.gz')
    print(data)
    test_image = []

    MIN_BOUND = -100.0
    MAX_BOUND = 400.0
    for i in range(len(data)):
        # data[i] = (data[i] - np.min(data[i])) / (np.max(data[i]) - np.min(data[i]))
        # test_image.append(transform.resize(nib.load(data[i]).get_data(), (256, 128, 96)))  # normalization data size
        test_image.append(transform.resize(sitk.GetArrayFromImage(sitk.ReadImage(data[i])).astype(np.float32), (96, 256, 128)))   # normalization data size
        # test_image.append(transform.resize(nib.load(data[i]).get_data(), (256, 256, 256)))  # normalization data size

    test_image = (test_image - np.min(test_image)) / (np.max(test_image) - np.min(test_image))
        # test_image[i] = (test_image[i] - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        # test_image[i][test_image[i] > 1] = 1
        # test_image[i][test_image[i] < 0] = 0
    image = (test_image - np.mean(test_image)) / (np.std(test_image))
    image = np.array(test_image)
    image = np.transpose(image, (0, 2, 3, 1))
    image = (image * 255).astype(np.float32)
    image = image[np.newaxis, :]
    image = np.transpose(image, (1, 2, 3, 4, 0))
    return image, data


# predict
def predict(test_data, model_path):

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

    imgs_test, data_file = read_test(test_data)

    print('***'*30)
    print('Loading saved weights...')
    print('***'*30)

    model = get_unet((256, 128, 96, 1))
    # model = get_unet((128, 128, 128, 1))
    # model = get_unet((256, 256, 256, 1))
    model.load_weights(model_path)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)

    imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
    # imgs_mask_test = imgs_mask_test.astype(np.uint8)
    imgs_mask_test = np.argmax(imgs_mask_test, axis=-1)
    imgs_mask_test = imgs_mask_test.astype(np.uint8)
    print("======")
    print(imgs_mask_test.shape)  # 256 128 96
    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)

    for i in range(len(imgs_mask_test)):
        # save_path = os.path.join(, 'verse' + str(f"{i}") + '_seg.nii.gz')
        out = sitk.GetImageFromArray(imgs_mask_test[i])
        sitk.WriteImage(out, 'pred_dir/{}_seg.nii.gz'.format(data_file[i].split('/')[-1].split(".")[0]))

    print('-'*30)
    print('Prediction finished')
    print('-'*30)


if __name__ == '__main__':

    print("loading dataset done\n")
    # test_data = np.load("x_test.npy").astype(np.float32)
    # test_data = "/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/Verse/Versedata/Verse2019/test_phase_1_public"
    test_data = "/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/Verse/Versedata/Verse2019/data_test/data"
    # test_data = "test_data"
    # read_test(test_data)
    print("loading model done\n")
    # model_to_load = "model/ResUnet.582.hdf5"
    model_to_load = "model/ResUnet.hdf5"

    predict(test_data, model_to_load)
    print("test finished")
