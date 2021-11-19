## 提取数字图片特征向量

在机器学习中有一种学习叫做手写数字识别，其主要功能就是让机器识别出图片中的数字，其步骤主要包括：图片特征提取、将特征值点阵转化为特征向量、进行模型训练。第一步便是提取图片中的特征提取。数据的预处理关系着后面模型的构建情况，所以，数据的处理也是机器学习中非常重要的一部分。下面我就说一下如何提取图片中的特征向量。
①图片灰度化：当我们拿到一种图片的时候，这张图片可能是多种颜色集合在一起的，而我们为了方便处理这张图片，我们首先会将这张图片灰度化。如果该图片已经是黑白两色的就可以省略此步骤。

②图片的二值化：就是将上面的数组化为0和1的形式，转化之前我们要设定一个阈值，大于这个阈值的像素点我们将其设置为1，小于这个阈值的像素点我们将其设置为0。

为什么要将二维的点阵转化成一维的特征向量? 这是因为在机器学习中，数据集的格式就是这样的，数据集的一个样例就是一个特征向量，对个样例组成一个训练集。转化为以为的特征向量是便于我们的使用。

输入：一个数字图片

输出：二值化后的数组、网络特征统计图

```python
from PIL import Image
import numpy as np


# 将二值化后的数组转化成网格特征统计图

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
img = Image.open("number3.jpg")
# 将图片化为32*32的
img = img.resize((32, 32))

# 二值化
img = img.point(lambda x: 1 if x > 120 else 0)
# 将图片转换为数组形式，元素为其像素的亮度值
img_array = np.asarray(img)
print(img_array)

#得到网格特征统计图
features_array = get_features(img_array)
print(features_array)

features_vector = features_array.reshape(features_array.shape[0] * features_array.shape[1])
print(features_vector)

```



## 施密特正交化

#### 施密特正交化几何解释

给定一组基α1,α2,...,αn,将其变换成另外一组正交基β1,β2,...,βn,使这两组基等价
 施密特正交化方法：



![img](https:////upload-images.jianshu.io/upload_images/21809199-77d4f98517a5fd31.png?imageMogr2/auto-orient/strip|imageView2/2/w/432/format/webp)



首先清除一个公式，两个向量α,β,那么α在β上的投影向量为

![img](https:////upload-images.jianshu.io/upload_images/21809199-30c1551b87531d1f.png?imageMogr2/auto-orient/strip|imageView2/2/w/52/format/webp)

如图红色部分即为投影部分



![img](https:////upload-images.jianshu.io/upload_images/21809199-bcfb5e82cb22a0fe?imageMogr2/auto-orient/strip|imageView2/2/w/521/format/webp)

则蓝色部分向量为

![img](https:////upload-images.jianshu.io/upload_images/21809199-830b6522f5010e96.png?imageMogr2/auto-orient/strip|imageView2/2/w/116/format/webp)

对应两个向量的施密特法则

![img](https:////upload-images.jianshu.io/upload_images/21809199-1f8aab1a68dd17cb.png?imageMogr2/auto-orient/strip|imageView2/2/w/177/format/webp)



可见蓝色向量为β2与β1是垂直的
 而当向量个数为3时，对应三维空间的几何解释如图



![img](https:////upload-images.jianshu.io/upload_images/21809199-d3459d5fd3477e00?imageMogr2/auto-orient/strip|imageView2/2/w/887/format/webp)

其中绿色的为需要正交的原始基αi（α1是红色的因为α1同时也是β1）
 将二维得到的β2平移到坐标原点出后则α3在xoy平面的投影即是

即α3在β1和β2上的投影组成的平行四边形的斜边，则得到的β3就是α3与该投影的向量差，即红色部分的β3,显然可以看出来β1,β2,β3是正交的。
 同样可以推广到三维以上的欧氏空间Rm,即施密特正交公式。

### 施密特正交化代码

施密特增强的流程便是通过计算原始图像的正交向量生成的，这些原始图像从不同类别中选取，作为黑箱模型标记。

输入：给定向量，即图像中的向量

输出：原始图像的正交向量

```python
import numpy as np
A = np.array([[1,1,0],[0,1,1],[1,0,1]],dtype=float)
Q = np.zeros_like(A)
m, n = Q.shape
cnt = 0
for a in A.T:
    u = np.copy(a)
    for i in range(0, cnt):
        u -= np.dot(np.dot(Q[:, i].T, a), Q[:, i]) # 减去待求向量在以求向量上的投影
    e = u / np.linalg.norm(u)  # 归一化
    Q[:, cnt] = e
    cnt += 1
print(Q)
```

------



## 雅克比迭代法

### 雅可比算法原理

已知：现有n元线性方程组，如何通过代码实现该方程组的有解的判定，以及自变量求解？

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200510203525966.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FjY2VwdGVkZGF5,size_16,color_FFFFFF,t_70#pic_center)

矩阵形式如下：

设方程组Ax=b的系数矩阵A非奇异，且主对角元素aii≠0(i=1,2,…,n),则可将A分裂成：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200510203540409.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FjY2VwdGVkZGF5,size_16,color_FFFFFF,t_70#pic_center)
Ax=b等价为矩阵形式的过程如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200510203549345.png#pic_center)

### 实现代码

输入：
输出：

```python
# -*- coding: utf-8 -*-

# Jacobi迭代法 输入系数矩阵mx、值矩阵mr、迭代次数n、误差c(以list模拟矩阵 行优先)

def Jacobi(mx, mr, n=100, c=0.0001):
    if len(mx) == len(mr):  # 若mx和mr长度相等则开始迭代 否则方程无解
        x = []  # 迭代初值 初始化为单行全0矩阵
        for i in range(len(mr)):
            x.append([0])
        count = 0  # 迭代次数计数
        while count < n:
            nx = []  # 保存单次迭代后的值的集合
            for i in range(len(x)):
                nxi = mr[i][0]
                for j in range(len(mx[i])):
                    if j != i:
                        nxi = nxi + (-mx[i][j]) * x[j][0]
                nxi = nxi / mx[i][i]
                nx.append([nxi])  # 迭代计算得到的下一个xi值
            lc = []  # 存储两次迭代结果之间的误差的集合
            for i in range(len(x)):
                lc.append(abs(x[i][0] - nx[i][0]))
            if max(lc) < c:
                return nx  # 当误差满足要求时 返回计算结果
            x = nx
            count = count + 1
        return False  # 若达到设定的迭代结果仍不满足精度要求 则方程无解
    else:
        return False


# 调用 Jacobi(mx,mr,n=100,c=0.001) 示例
mx = [[8, -3, 2], [4, 11, -1], [6, 3, 12]]

mr = [[20], [33], [36]]
print(Jacobi(mx, mr, 100, 0.00001))
```

