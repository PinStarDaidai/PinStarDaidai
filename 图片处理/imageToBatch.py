import numpy as np
from PIL import Image
import operator
from os import listdir
import sys
import pickle
import random

data = {}
list1 = []
list2 = []
list3 = []


# 将图片转化为32*32的三通道图片
def img_tra():
    for k in range(0, num):
        currentpath = folder + "/" + imglist[k]
        im = Image.open(currentpath)
        # width=im.size[0]
        # height=im.size[1]
        x_s = 32
        y_s = 32
        out = im.resize((x_s, y_s), Image.ANTIALIAS)
        out.save(folder_ad + "/" + str(imglist[k]))


def addWord(theIndex, word, adder):
    theIndex.setdefault(word, []).append(adder)


# 图片存储格式
def seplabel(fname):
    filestr = fname.split(".")[0]
    label = int(filestr.split("_")[0])
    return fname


# 我们将点操作与写文件写在一个函数mkcf()函数中
def mkcf():
    global data
    global list1
    global list2
    global list3

    for k in range(0, num):  # 图片数量
        currentpath = folder_ad + "/" + imglist[k]  # 现在是第几张图
        im = Image.open(currentpath)  # 打开图片
        # im = im.convert("RGB")
        im = im.convert('L')
        with open(binpath, 'a') as f:  # a是函数的设定，打开图片，添加信息到list1
            for i in range(0, 32):
                for j in range(0, 32):
                    # print(im)
                    # print(im.mode)
                    # print(im.getpixel((0, 0)))
                    cl = im.getpixel((i, j))
                    print(cl)
                    list1.append(cl[0])

            for i in range(0, 32):
                for j in range(0, 32):
                    cl = im.getpixel((i, j))
                    # with open(binpath, 'a') as f:
                    # mid=str(cl[1])
                    # f.write(mid)
                    list1.append(cl[1])

            for i in range(0, 32):
                for j in range(0, 32):
                    cl = im.getpixel((i, j))
                    list1.append(cl[2])

        list2.append(list1)
        list1 = []
        f.close()
        print("image" + str(k + 1) + "saved.")

        list3.append(imglist[k].encode('utf-8'))

    arr2 = np.array(list2, dtype=np.uint8)
    data['batch_label'.encode('utf-8')] = 'batch_1'.encode('utf-8')  # batch label包的名字
    data.setdefault('labels'.encode('utf-8'), label)  # lable 特征
    data.setdefault('data'.encode('utf-8'), arr2)  # data 图像
    data.setdefault('filenames'.encode('utf-8'), list3)  # filename文件名

    output = open(binpath, 'wb')
    pickle.dump(data, output)
    output.close()


# folder = r"D:\lhh\python\demo\图片制作数据集\left50s"
name = 'test'
folder_ad = r"D:\lhh\python\demo\图片制作数据集\shujuji\size54_50pics\{}".format(name)
imglist = listdir(folder_ad)
num = len(imglist)
# img_tra()
label = []
for i in range(0, num):
    label.append(seplabel(imglist[i]))
# print(label)
binpath = r"D:\lhh\python\demo\图片制作数据集\{}batch".format(name)
mkcf()

