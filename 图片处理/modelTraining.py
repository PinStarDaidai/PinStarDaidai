# %% md

# 目录
## 1. [CIFAR10数据集介绍](#1.CIFAR10数据集介绍)
## 2. [VGG19起源与介绍](#2.VGG19起源与介绍)
# 2.1.[VGGNet简介](  # 2.1.VGGNet简介)
#     2.2.[VGG16与VGG19的对比](  # 2.2.VGG16与VGG19的对比)
#         ## 3. [利用Pytorch构建VGG19模型](#3.利用Pytorch构建VGG19模型)
#         3.1.[数据预处理](  # 3.1.数据预处理)
#             3.2.[搭建VGG19](  # 3.2.搭建VGG19)
#                 3.3.[训练VGG19](  # 3.3.训练VGG19)
#                     ## 4. [总结](#4.总结)
#
#                     # %% md
#
#                     # 1.CIFAR10数据集介绍
#                     CIFAR - 10
# 数据集由10个类别共60000张32×32
# 的彩色图像组成，每个类别有6000张图像。有50000张训练图像和10000张测试图像。
#
# 数据集分为五个训练批次和一个测试批次，每个批次有10000张图像。测试批次包含来自每个类别的恰好1000个随机选择的图像。但一些训练批次中某个类别的图像张数要比其它类别多，总体来说，五个训练批次一共包含来自每个类别的正好5000张图像。
#
# 以下是数据集中的类别，以及来自每个类别的10个随机图像：
# ![png](https: // i.loli.net / 2018 / 02 / 21 / 5
# a8cd1b0076ff.png)

# %% md

# 2.VGG19起源与介绍

# %% md

## 2.1. VGGNet简介
# VGGNet是牛津大学计算机视觉组（Visual
# Geometry
# Group）和Google
# DeepMind公司的研究员一起研发的卷积神经网络。VGGNet探索了卷积神经网络的深度与其性能之间的关系，通过反复的使用$3\times3$的小型卷积核和$2\times2$的最大池化层，VGGNet成功地构筑了16～19
# 层深的卷积神经网络。
#
# VGG19的网络结构如图所示：
# ![png](https: // raw.githubusercontent.com / shiyadong123 / Myimage / master / 20170816092916647.
# png)

# %% md

## 2.2. VGG16与VGG19的对比
# 相较于VGG16，VGG19网络增加了3个卷积层，其余的部分是相同的。
# ![png](https: // github.com / shiyadong123 / Myimage / blob / master / 20190217165325787
# _meitu_1.png?raw = true)

# %% md

# 3. 利用Pytorch构建VGG19模型

# %% md

## 3.1. 数据预处理
# 在搭建VGG19网络结构之前，首先要对数据进行一些预处理。

# %%

'''
import json
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset,DataLoader
'''

# %%

# 导入必要的库
import torch
import torchvision
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as trans
import time
import matplotlib.pyplot as plt
import numpy as np

# %%

from torch.utils.data import Dataset
import json


class EYEIMG(Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(EYEIMG, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # 如果是训练则加载训练集，如果是测试则加载测试集
        if self.train:
            file_annotation = root + '/annotations/eye_train.json'
            img_folder = root + '/data/train_eye/'
        else:
            file_annotation = root + '/annotations/eye_test.json'
            img_folder = root + '/data/test_eye/'
        fp = open(file_annotation, 'r')
        data_dict = json.load(fp)

        # 如果图像数和标签数不匹配说明数据集标注生成有问题，报错提示
        assert len(data_dict['images']) == len(data_dict['categories'])
        num_data = len(data_dict['images'])

        self.filenames = []
        self.labels = []
        self.img_folder = img_folder
        for i in range(num_data):
            # self.filenames.append(data_dict['images'][i].split('~')[0])
            # self.labels.append(data_dict['categories'][i].split('~')[0])
            self.filenames.append(data_dict['images'][i])
            self.labels.append(data_dict['categories'][i])

    def __getitem__(self, index):
        img_name = self.img_folder + self.filenames[index]
        label = self.labels[index]

        img = plt.imread(img_name)
        img = self.transform(img)  # 可以根据指定的转化形式对数据集进行转换

        # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
        return img, label

    def __len__(self):
        return len(self.filenames)


# %% md

# 设置batch的大小为10，迭代轮数为20，优化算法的初始学习率设为0
# .001，损失函数定义为交叉熵损失，并定义是否在GPU上运行。

# %%

# 超参数
BATCH_SIZE = 10
nepochs = 40
LR = 0.001

# 定义损失函数为交叉熵损失 loss_func
loss_func = nn.CrossEntropyLoss()

# 可以在GPU或者CPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %% md

# 数据集的预处理分为三步： < br >
# 第一步：求出CIFAR10的输入图片各channel的均值
# `mean`
# 和标准差
# `std`，便于后续进行标准化。 < br >
# + 为了更方便求出数据集的
# `mean`
# 和
# `std`，封装了一个
# `get_mean_std`
# 函数，其思想主要是随机从数据集采样，直接调用
# `numpy`
# 的方法返回数据集样本的均值和方差


# %%

# 定义一个简单函数用来求数据集的mean和std
def get_mean_std(dataset, ratio=0.01):
    # dataloader = DataLoader(dataset, batch_size=int(len(dataset)*ratio), shuffle=True,)
    dataloader = DataLoader(dataset, batch_size=int(len(dataset)), shuffle=True, )
    # 加载数据集
    train = iter(dataloader).next()[0]
    # 求mean
    mean = np.mean(train.numpy(), axis=(0, 1, 2, 3))
    # 求std
    std = np.std(train.numpy(), axis=(0, 1, 2, 3))
    return mean, std


# 加载训练集数据，求出训练集的mean和std，用于标准化
train_set = EYEIMG(root=r'D:\lhh\python\demo\图片制作数据集\shujuji',  # 数据集保存路径
                   train=True,
                   transform=trans.ToTensor())

# 加载测试集数据，求出测试集的mean与std，用于标准化
test_set = EYEIMG(root=r'D:\lhh\python\demo\图片制作数据集\shujuji',  # 测试集保存路径
                  train=False,
                  transform=trans.ToTensor())

# 得到mean与std
mean_train, std_train = get_mean_std(train_set)
mean_test, std_test = get_mean_std(test_set)

# 打印mean与std
print(mean_train, std_train, mean_test, std_test)

# %% md

# 第二步：对训练集进行数据增强，对数据集中的图片进行随机水平翻转与随机裁剪，然后将其变换成Tensor，最后进行标准化。

# %%

# 训练集样本数
n_train_samples = 40

# 如果是多进程需要加一个main函数，否则会报错

# 加载训练集并进行数据增强
train_set = EYEIMG(root=r'D:\lhh\python\demo\图片制作数据集\shujuji',  # 数据集保存路径
                   transform=trans.Compose([
                       trans.ToPILImage(),
                       trans.RandomHorizontalFlip(),  # 随机水平翻转
                       # trans.RandomCrop([64,64], padding=4), # 随机裁剪
                       trans.ToTensor(),
                       trans.Normalize(mean_train, std_train)  # 标准化
                   ]))  # 转为Tensor

# 将训练集按batch大小进行分割
train_dl = DataLoader(train_set,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers=0)  # 多进程
# print('train_dl:', train_dl)

# %% md

# 第三步：与第二步类似，对测试集执行同样的操作，转换为Tensor并进行标准化。

# %%

# 测试集
test_set = EYEIMG(root='D:\lhh\python\demo\图片制作数据集\shujuji',  # 数据集保存路径
                  transform=trans.Compose([
                      trans.ToPILImage(),
                      trans.ToTensor(),
                      trans.Normalize(mean_test, std_test)
                  ]))

test_dl = DataLoader(test_set,
                     batch_size=BATCH_SIZE,
                     num_workers=0)  # 多进程

# %% md

# 输出5张数据集图片：

# %%

# 数据集展示
# % matplotlib
# inline

# 为了更好的显示部分图片，重新设置了一个 batch_size=5
train_data = DataLoader(train_set,
                        batch_size=5,
                        shuffle=True,
                        )


def imshow(img):
    img = img / 2 + 0.5  # 原始数值为[-1,1],，需转换到[0,1]
    npimg = img.numpy()
    # 设置图片大小
    plt.figure(figsize=(64, 64))
    # 输出图片
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # 去掉横纵坐标轴
    plt.xticks(())
    plt.yticks(())


# 从 train_data 中获得照片
dataiter = iter(train_data)
images, labels = dataiter.next()

# 图片展示
imshow(torchvision.utils.make_grid(images))
# 打印标签
classes = ('0', '1', '2', '3')
print(labels)
print('            '.join('%5s' % classes[int(labels[j].split('_')[0])] for j in range(5)))


# %% md

# 定义训练模型用的辅助函数：

# %%

# 定义训练的辅助函数，其中包括误差 error 与正确率 accuracy
def eval(model, loss_func, dataloader):
    model.eval()
    # 损失与正确率初始值
    loss, accuracy = 0, 0

    # torch.no_grad显示地告诉pytorch，前向传播的时候不需要存储计算图
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y

            # 模型输出
            logits = model(batch_x)
            batch_y = [int(i.split('_')[0]) for i in batch_y]
            batch_y = torch.Tensor(batch_y).long()
            # print(batch_y)
            # 误差
            error = loss_func(logits, batch_y)
            # 损失
            loss += error.item()
            # 预测值
            probs, pred_y = logits.data.max(dim=1)
            # 正确率
            accuracy += (pred_y == batch_y.data).float().sum() / batch_y.size(0)
    # 最终损失
    loss /= len(dataloader)
    # 最终正确率
    accuracy = accuracy * 100.0 / len(dataloader)
    return loss, accuracy


def train_epoch(model, loss_func, optimizer, dataloader):
    model.train()
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        # print('111', batch_x)
        # print('222', batch_x.to('cpu'))
        # print(device)
        # print(batch_x[0])
        # batch_x = batch_x.to(device)
        batch_x = batch_x.cpu()
        batch_y = [int(i.split('_')[0]) for i in batch_y]
        batch_y = torch.Tensor(batch_y).long()
        # print(batch_y)
        # batch_x = torch.tensor(batch_x)
        # batch_y = batch_y[0.cpu()
        # 权值初始化

        # 模型输出
        # print(batch_x)
        # for i in batch_x:
        #     print(i[0][0])
        # for i in batch_y:
        # print(batch_x.size)
        #     print(i[0][0])
        # print(i[0][0][0])
        # print(i[0][0][1])
        logits = model(batch_x)
        # print(logits)
        # 误差计算
        error = loss_func(logits, batch_y)
        # 误差梯度反向传播
        error.backward()
        # 参数更新
        optimizer.step()


# %% md

# ## 3.2. 搭建VGG19
# ![png](https: // github.com / shiyadong123 / Myimage / blob / master / 68747470733
# a2f2f6c6968616e2e6d652f6173736574732f696d616765732f7667672d6865726f2d636f7665722e6a7067.jpg?raw = true)
# 根据上图的VGG19网络结构，开始正式搭建VGG19模型，为了方便起见，先定义卷积层


# %%

# 定义卷积层，在VGGNet中，均使用3x3的卷积核
def conv3x3(in_features, out_features):
    return nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)


# %% md

# VGG19一共包含19个隐藏层，其中16个卷积层，3
# 个全连接层，具体结构如下：
#
# 卷积层1：64
# 个3×3
# 的卷积核 < br >
# 卷积层2：64
# 个3×3
# 的卷积核 < br >
# 池化层1：2×2
# 最大池化 < br >
#
# 卷积层3：128
# 个3×3
# 的卷积核 < br >
# 卷积层4：128
# 个3×3
# 的卷积核 < br >
# 池化层2：2×2
# 最大池化 < br >
#
# 卷积层5：256
# 个3×3
# 的卷积核 < br >
# 卷积层6：256
# 个3×3
# 的卷积核 < br >
# 卷积层7：256
# 个3×3
# 的卷积核 < br >
# 卷积层8：256
# 个3×3
# 的卷积核 < br >
# 池化层3：2×2
# 最大池化 < br >
#
# 卷积层9：512
# 个3×3
# 的卷积核 < br >
# 卷积层10：512
# 个3×3
# 的卷积核 < br >
# 卷积层11：512
# 个3×3
# 的卷积核 < br >
# 卷积层12：512
# 个3×3
# 的卷积核 < br >
# 池化层4：2×2
# 最大池化 < br >
#
# 卷积层13：512
# 个3×3
# 的卷积核 < br >
# 卷积层14：512
# 个3×3
# 的卷积核 < br >
# 卷积层15：512
# 个3×3
# 的卷积核 < br >
# 卷积层16：512
# 个3×3
# 的卷积核 < br >
# 池化层5：2×2
# 最大池化 < br >
#
# 全连接层1：4096
# 个神经元 < br >
# 全连接层2：4096
# 个神经元 < br >
# Softmax层：1000
# 个神经元（对应目标类别为1000种的分类问题）

# %%

# 搭建VGG19，除了卷积层外，还包括2个全连接层，1个softmax层
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # 1.卷积层1
            conv3x3(1, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 2.卷积层2
            conv3x3(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 池化层1
            # 3.卷积层3
            conv3x3(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 4.卷积层4
            conv3x3(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 池化层2
            # 5.卷积层5
            conv3x3(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 6.卷积层6
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 7.卷积层7
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 8.卷积层8
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 池化层3
            # 9.卷积层9
            conv3x3(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 10.卷积层10
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 11.卷积层11
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 12.卷积层12
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 池化层4
            # 13.卷积层13
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 14.卷积层14
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 15.卷积层15
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # 16.卷积层16
            conv3x3(512, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 池化层5
        )

        self.classifier = nn.Sequential(
            # 17.全连接层_1
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 18.全连接层_2
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # 19.softmax层
            nn.Linear(4096, 10),  # 最后通过softmax层，输出10个类别
        )

    def forward(self, x):
        # print(x)
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


# %%

vgg19 = VGG().to(device)
# 可以通过打印vgg19观察具体的网络结构
print(vgg19)

# %% md

# 使用Adam方法进行优化处理

# %%

optimizer = torch.optim.Adam(vgg19.parameters(), lr=LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)
learn_history = []

# %% md

## 3.3. 训练VGG19

# %%

print('开始训练VGG19……')

for epoch in range(nepochs):
    # 训练开始时间
    since = time.time()
    # 利用定义的训练辅助函数
    train_epoch(vgg19, loss_func, optimizer, train_dl)

    # 每训练5轮输出一次结果
    if (epoch) % 5 == 0:
        tr_loss, tr_acc = eval(vgg19, loss_func, train_dl)
        te_loss, te_acc = eval(vgg19, loss_func, test_dl)
        learn_history.append((tr_loss, tr_acc, te_loss, te_acc))
        # 完成一批次训练的结束时间
        now = time.time()
        # print('[%3d/%d, %.0f seconds]|\t 训练误差: %.1e, 训练正确率: %.2f\t |\t 测试误差: %.1e, 测试正确率: %.2f' % (
        #     epoch + 1, nepochs, now - since, tr_loss, tr_acc, te_loss, te_acc))
        print('[{}/{}, {} seconds]|\t 训练误差: {}, 训练正确率: {}\t |\t 测试误差:{}, 测试正确率: {}'.format(
            epoch + 1, nepochs, '%.2f' % (now - since), '%.2f' % (tr_loss), '%.2f' % (tr_acc), '%.2f' % (te_loss),
            '%.2f' % (te_acc)))

# %% md

# 根据输出的结果，我们可以看出，在第6轮迭代的时候，分类正确率就已经接近70 %，在训练轮次增加后，正确率又有了明显的提高，训练完30轮后，测试集的正确率达到88
# .12 %。至此，我们完成了对VGG19的训练。

# %% md

# 4.总结

# %% md

# 在本案例中，我们首先了解了CIFAR10数据集，接着介绍了VGG19的基本特点和网络结构，最后利用Pytorch建立了VGG19模型对CIFAR10数据集进行了图像分类，最终模型在测试集上的分类正确率为88 %。CIFAR10是一个相当常见并且经典的数据集，在深度学习中被广泛拿来进行各种网络结构模型的分类测试，大家可以再尝试建立别的卷积神经网络模型，或许会得到更优的结果！

# %%

