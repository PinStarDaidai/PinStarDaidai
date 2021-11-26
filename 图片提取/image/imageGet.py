from PIL import Image
import numpy as np
#该方法为提取数字图片的特征向量，下面说说如何提取图片中的特征向量

#将二值化的数组转化为网络特征值
def get_features(array):
    # 拿到数组的高度和宽度
    h, w = array.shape
    data = []
    for x in range(0, w / 4):
        offset_y = x * 4
        temp = []
        for y in range(0, h / 4):
            offset_x = y * 4
            # 统计每个区域的1的值
            temp.append(sum(sum(array[0 + offset_y:4 + offset_y, 0 + offset_x:4 + offset_x])))
        data.append(temp)
    return np.asarray(data)


# 打开一张图片
#图片样式，一个黑白数字图片
img = Image.open("number3.jpg")

#为了统一，将所有得到的图片都化为32*32处理
img = img.resize((32, 32))
#二值化操作，在此之前要先灰度化处理，即把所有彩色图片都转化为黑白图片
#二值化就是将数据转化为0，1数组形式，设定一个阈值，大于这个阈值的像素点设为1，小于该阈值设为0
img = img.point(lambda x: 1 if x > 120 else 0)
#用lambda方法得到图片的二值化输出，再将二值化输出转化为网络特征统计图

# 将图片转换为数组形式，元素为其像素的亮度值
img_array = np.asarray(img)
print(img_array)

features_array = get_features(img_array)
print(features_array)

features_vector = features_array.reshape(features_array.shape[0] * features_array.shape[1])
print(features_vector)
