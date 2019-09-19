# -*- coding: utf-8 -*-

"""
@version: v2.0
@author: Lxg
@time: 2019/08/12
@email: lvxiaogang0428@163.com
@function: read npy file to .nii.gz data  & .nii.gz visualization
"""

import SimpleITK as sitk
import numpy as np
import cv2
import os
from glob import glob
import nibabel as nib
from skimage import io, transform


# read test data
def read_test(filename):
    # data = glob(filename + '/*.nii.gz')
    label = glob(filename + "/*_seg.nii.gz")
    # print(data)
    print(label)
    train_image = []
    # for i in range(len(data[:10])):
    for i in range(len(label[:])):
        train_image.append(sitk.GetArrayFromImage(sitk.ReadImage(label[i])))  # normalization data size

    # add windows
    # test_image = (test_image - np.min(test_image)) / (np.max(test_image) - np.min(test_image))
    # image = (test_image - np.mean(test_image)) / (np.std(test_image))
    image = np.array(train_image)

    # return image, data
    return image, label


def Visualization(data_path):

    # read to data
    # X_patch = np.load(data_path).astype(np.float32)
    X_patch = np.load(data_path).astype(np.uint8)
    for i in range(len(X_patch)):
        visual_path = "visual"
        save_path = os.path.join(visual_path, 'Gen' + str(f"{i}") + '.nii.gz')
        out = sitk.GetImageFromArray(X_patch[i])
        sitk.WriteImage(out, save_path)

    return


def visual(path):

    # sitk read data
    image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image)
    # get data size
    z, y, x = image.shape
    for i in range(z):
        cv2.imwrite("vis/{}.png".format(i), image[i, :, :])


def visual0nii(path):

    # sitk read data
    image_all = []
    imgs_train, data_file = read_test(path)
    for i in range(len(imgs_train)):
        # image_all.append(sitk.GetArrayFromImage(sitk.ReadImage(data_file[i])))
        image_all.append(transform.resize(sitk.GetArrayFromImage(sitk.ReadImage(data_file[i])).astype(np.uint8), (64, 128, 96)))

    image_all = np.array(image_all)
    image_all = (image_all * 255).astype(np.uint8)  #
    image_all = np.transpose(image_all, (0, 2, 3, 1))
    for j in range(len(image_all)):
        # save_path = os.path.join(, 'verse' + str(f"{i}") + '_seg.nii.gz')
        out = sitk.GetImageFromArray(image_all[j])
        sitk.WriteImage(out, 'seg_nii/{}.nii.gz'.format(data_file[j].split('/')[-1].split(".")[0]))


def visual01(path):
    # sitk read data
    imgs_train, data_file = read_test(path)
    for i in range(len(imgs_train)):
        image = sitk.ReadImage(data_file[i])
        image = sitk.GetArrayFromImage(image)
        # image = transform.resize(sitk.GetArrayFromImage(image), (96, 96, 96))
        # make subdir
        # file = data_file[i].split('/')[-1].split(".")[0]
        # vis = "VIS/"
        # vis_subdir = os.mkdir(vis + file)
        # get data size
        z, y, x = image.shape
        for j in range(z):
            # cv2.imwrite("VIS/{}_{}.png".format(data_file[i].split('/')[-1].split(".")[0], j), image[j, :, :])  # none subdirectory  VIS/{}/{}.png is error
            # cv2.imwrite("VIS_seg/{}_{}.png".format(data_file[i].split('/')[-1].split(".")[0], j), image[j, :, :])  # none subdirectory  VIS/{}/{}.png is error
            cv2.imwrite("VIS_seg01/{}_{}.png".format(data_file[i].split('/')[-1].split(".")[0], j), image[j, :, :])  # none subdirectory  VIS/{}/{}.png is error
            # cv2.imwrite("VIS01/{}_{}.png".format(data_file[i].split('/')[-1].split(".")[0], j), image[j, :, :])  # none subdirectory  VIS/{}/{}.png is error
            # cv2.imwrite("vis_subdir/{}.png".format(j), image[j, :, :])  # none subdirectory  VIS/{}/{}.png is error
            # io.imsave("VIS/{}/{}.png".format(data_file[i].split('/')[-1].split(".")[0], j), image[j, :, :])

    return imgs_train


if __name__ == "__main__":

    # npy data
    # print("read _.npy file to _.nii.gz file")
    # npy_data = "y_training.npy"
    # Visualization(npy_data)
    # print("read _.nii.gz file to _.png file")
    # # nii_data = "visual/Gen1.nii.gz"
    # nii_data = "/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/Verse/Versedata/Verse2019/test_phase_1_public/verse078.nii.gz"
    # visual(nii_data)
    # print("**end**" * 10)

    print("01 read _.nii.gz file to _.png file")
    train_data = "/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/Verse/Versedata/Verse2019/data"
    visual0nii(train_data)
    # visual01(train_data)
    print("*********End***********")

    # test normalization resize is (96, 96, 96)
    #test_data = "/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/Verse/Versedata/Verse2019/test_phase_1_public"
    # train_seg_data = "/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/Verse/Versedata/Verse2019/data"
    # train_seg_data = "conv_label/iaiot"
    # # visual01(test_data)
    # visual01(train_seg_data)



