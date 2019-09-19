# -*- encoding: utf-8 -*-
import numpy as np
import nibabel as nib
import os


def read_data(data_path):
    data_all = []
    label_all = []

    for label in os.listdir(data_path):
        if "_seg.nii.gz" in label:
            # print("======")
            # print(label)
            for data in os.listdir(data_path):
                if data[:-7] == label[:-11]:
                    # print(data)
                    # ono-to-one
                    data_all.append(data)
            label_all.append(label)

    data_read = []
    label_read = []
    a_list = []
    b_list = []
    c_list = []
    for i in range(len(data_all)):
        data_read.append(nib.load(os.path.join(data_path, data_all[i])).get_data())
        label_read.append(nib.load(os.path.join(data_path, label_all[i])).get_data())
        # print("data.shape", data_read[i].shape)
        # print("label.shape", label_read[i].shape)
        # print("next data&label")
        print("******verse data compare******")
        print("num{}, data_name:{}: data shape:{}, label shape:{}; Pixel -- data_max:{}, data_min:{}, "
              "label_max:{}, label_min:{} "
              .format(i, data_all[i].split('/')[-1].split(".")[0],
                      data_read[i].shape, label_read[i].shape,
                      data_read[i].max(),  data_read[i].min(),
                      label_read[i].max(),  label_read[i].min()))
        print("===***==="*10)
        # if data size != label.size print data
        a, b, c = data_read[i].shape
        la, lb, lc = label_read[i].shape
        if a != la or b !=lb or c !=lc:
            print('data different label size :{}'.format(data_all[i].split('/')[-1].split(".")[0]))

        if label_read[i].max() == 25:
            print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&label is 25 data'.format(data_all[i].split('/')[-1].split(".")[0]))



    return data_read, label_read


if __name__ == "__main__":
    # data_path = '/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/Verse/Versedata/Verse2019/data'
    data_path = '/media/tx-deepocean/393521a8-72ec-4c9d-9edb-744aa979e170/data/lxg/Verse/Versedata/Verse2019/data'
    print("read_data start ...")
    read_data(data_path)
    print("read_data end ...")
