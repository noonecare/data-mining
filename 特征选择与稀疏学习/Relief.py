# coding: utf-8
import numpy as np


"""
实现 Relief 算法
"""


class Relief:
    def __init__(self, sample: np.ndarray):
        # 包含 feature 和 class 的样本，最后一列是 class
        self.sample = sample
        # 距离函数, 默认是 L_1 范数
        self.diff = np.abs
        # 记录了任意两个 sample 点之间的距离
        self.diff_matrix = None
        self.is_same_class = None

    # 计算出所有样本点之间的距离, 就用简单的欧式距离
    def init_diff_matrix(self):
        m = self.sample.shape[0]
        dist = []
        # 后续可以减少一半的运算时间开销
        for i in range(m):
            for j in range(m):
                dist.append(np.sqrt(np.sum(self.sample[i, : -1] - self.sample[j, -1])))
        self.diff_matrix = np.array(dict).reshape(m, m)
        self.is_same_class = self.sample[:, -1] + self.sample[:, -1].T

    def find_nt(self):
        pass

    def find_nm(self):
        pass


    def delta(self):
        pass


if __name__ == '__main__':
    a = np.zeros(3)
    b = a + 1
    print(np.abs(b - a))
