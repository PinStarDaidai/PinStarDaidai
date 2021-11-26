#%%

import torch
import torchvision
import numpy as np
import cv2
import os
import json

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

anno_loc = r'D:\lhh\python\demo\图片制作数据集\shujuji\annotations'

#判断文件夹是否存在，不存在的话创建文件夹
if os.path.exists(anno_loc) == False:
    os.mkdir(anno_loc)

#用于存放图片文件名及标注
train_filenames = []
train_annotations = []

test_filenames = []
test_annotations= []

#训练集有五个批次，每个批次10000个图片，测试集有10000张图片
def eye_annotations(file_dir):
    print('creat train_img annotations')
    for i in range(1,5):
        data_name = file_dir + '/' + 'data_batch_' + str(i)
        data_dict = unpickle(data_name)
        print(data_name + ' is processing')
        for j in range(10):
            img_name = str(data_dict[b'labels'][j])
            img_annotations = data_dict[b'labels'][j]
            train_filenames.append(img_name)
            train_annotations.append(img_annotations)
        print(data_name + ' is done')

    test_data_name = file_dir + '/test_batch'
    print(test_data_name + ' is processing')
    test_dict = unpickle(test_data_name)

    for m in range(10):
        testimg_name = str(test_dict[b'labels'][m])
        testimg_annotations = test_dict[b'labels'][m]     #str(test_dict[b'labels'][m])    test_dict[b'labels'][m]
        test_filenames.append(testimg_name)
        test_annotations.append(testimg_annotations)

    print(test_data_name + ' is done')
    print('Finish file processing')


if __name__ == '__main__':

    file_dir = 'D:\lhh\python\demo\图片制作数据集\shujuji'
    eye_annotations(file_dir)

    train_annot_dict = {
        'images': train_filenames,
        'categories': train_annotations
    }
    test_annot_dict = {
        'images':test_filenames,
        'categories':test_annotations
    }
    # print(annotation)

    train_json = json.dumps(train_annot_dict)
    print(train_json)
    train_file = open(r'D:\lhh\python\demo\图片制作数据集\shujuji\annotations\eye_train.json', 'w')
    train_file.write(train_json)
    train_file.close()

    test_json =json.dumps(test_annot_dict)
    test_file = open(r'D:\lhh\python\demo\图片制作数据集\shujuji\annotations\eye_test.json','w')
    test_file.write(test_json)
    test_file.close()
    print('annotations have writen to json file')

#%%



