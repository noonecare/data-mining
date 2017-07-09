# coding: utf-8
import numpy as np


class BP:
    def __init__(self, sample: np.ndarray, d: int, q: int, eta: float, max_iteration=1000):
        """
        :param sample: 样本函数， 前 d 列是特征列，后 q 列是输出的 class 
        :param d: 特征的维度
        :param q: 隐含层神经元的个数
        :param eta: 训练的步长
        """
        self.max_iteration = max_iteration
        self.eta = eta
        self.sample = sample
        self.d = d
        self.l = self.sample.shape[1] - self.d
        self.q = q
        # 隐含层到输出层的权重
        self.w = np.random.randint(0, 100, self.q * self.l).reshape(self.q, self.l) / 100
        # 隐含层到输出层的阈值
        self.theta = np.random.randint(0, 100, self.l) / 100
        # 输入层到隐含层的权重
        self.v = np.random.randint(0, 100, self.d * self.q).reshape(self.d, self.q) / 100
        # 输出层到隐含层的阈值
        self.gamma = np.random.randint(0, 100, self.q) / 100

        self.alpha = np.zeros(self.q)
        self.beta = np.zeros(self.l)
        self.iteration = 0

    def update(self, x_k, y_k):
        predict_y_k, b = self.predict(x_k)
        g = predict_y_k * (1 - predict_y_k) * (y_k - predict_y_k)
        e = b * (1 - b) * np.dot(self.w, g.reshape(len(g),))

        # 更新权重和阈值
        self.w += self.eta * np.dot(b.reshape(len(b), 1), g.reshape(1, len(g)))
        self.v += self.eta * np.dot(x_k.reshape(len(x_k), 1), e.reshape(1, len(e)))
        self.theta += - self.eta * g
        self.gamma += - self.eta * e
        self.iteration += 1

    def update_per_epoch(self):
        for sample_point in self.sample:
            x_k = sample_point[: self.d]
            y_k = sample_point[self.d: ]
            self.update(x_k, y_k)

    @staticmethod
    def sigmoid(x):
        return np.array([1 / (1 + value) for value in np.exp(-x)])

    # 暂时生硬地指定迭代次数
    def can_stop(self):
        return self.iteration > self.max_iteration

    def train(self):
        while self.can_stop() is False:
            self.update_per_epoch()

    def predict(self, x_k):
        # 更新 alpha 的值
        self.alpha = np.dot(x_k, self.v)
        # 更新 b 的值
        b = self.sigmoid(self.alpha)
        # 更新 beta 的值
        self.beta = np.dot(b, self.w)
        # 更新 y 的值
        y = self.sigmoid(self.beta)
        return y, b


if __name__ == '__main__':
    pass
