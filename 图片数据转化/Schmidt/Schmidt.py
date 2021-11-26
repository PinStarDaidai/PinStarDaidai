import numpy as np
#施密特正交化方法
#给定一组基向量，转为另外一组正交基
#给定的输入[1,1,0],[0,1,1],[1,0,1]]，给定的输入应为图像中的得来的向量数据
A = np.array([[1,1,0],[0,1,1],[1,0,1]],dtype=float)
Q = np.zeros_like(A)
m, n = Q.shape
cnt = 0
#对输入的向量构成的矩阵A进行如下处理
for a in A.T:
    u = np.copy(a)
    for i in range(0, cnt):
        u -= np.dot(np.dot(Q[:, i].T, a), Q[:, i]) # 减去待求向量在以求向量上的投影
    e = u / np.linalg.norm(u)  # 归一化
    Q[:, cnt] = e
    cnt += 1
#输出原始图像的正交向量
print(Q)