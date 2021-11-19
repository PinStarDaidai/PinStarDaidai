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