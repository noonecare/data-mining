# coding: utf-8
import pandas as pd
import numpy as np


"""
为了避免其他属性携带的信息被训练集中未出现的属性值抹去，在估计概率值时通常需要进行平滑，常用拉普拉斯修正，具体来说，令 N 表示训练集 D 中可能的类别数， N_i 表示第 i 个属性可能的取值数

P^{^}(c) = \frac {|D_c + 1|} {|D| + N}
P^{^}(x_i | c) = \frac {|D_{c, x_i}| + 1} {|D_c| + N_i}
"""

class LaplaceModifiedBayesianClassifier:
    """
    简单实现 Laplace 修正的 BayesianClassifier， Laplace 修正主要是为了防止下溢， 下溢指的是有的属性的条件概率为零，导致其他属性对于判别的影响完全消失。
    
    Bayesian Classifier 的实现主要为：
    求 类 的概率
    求 属性值对于类的 条件概率
    几个概率值累乘，属于哪个类的概率高，就认为样本点属于哪个类
    """

    def __init__(self, features: np.ndarray, classes: np.ndarray) -> None:
        """
        :param features: 二维的 np.ndarray，表示样本
        :param classes: 一维的 np.ndarray, 表示类别  
        """
        super().__init__()
        self.attributes_classes_probability = {}
        self.features = features
        self.classes = classes
        # m 个样本点，每个样本点有 n 个特征
        self.m, self.n = self.features.shape


    @staticmethod
    def laplace_modified_condition_probability(x: np.ndarray):
        distinct_values = np.unique(x)
        length = len(x)
        length_1 = len(distinct_values)
        frequency = [(value, (len(x[x == value]) + 1) / (length + length_1) for value in distinct_values]
        return frequency

    def calculate_class_probability(self):
        self.classes_probability = self.laplace_modified_condition_probability(self.classes)

    def calculate_attribute_probability(self):
        # attrs_to_classes 把特征和类属放在同一个矩阵中
        attrs_to_classes = np.zeros((self.m, self.n + 1))
        attrs_to_classes[:, : -1] = self.features
        attrs_to_classes[:, -1] = self.classes
        # 对于每个 clazz(class 是 python 的保留字，所以我用 clazz 代替)， 统计属性对于 clazz 的条件概率
        for clazz in self.classes_probability:
            data_belong_to_clazz = attrs_to_classes[attrs_to_classes[:,-1] == clazz]
            self.attributes_classes_probability.update({clazz: []})
            # 统计每个特征的类条件概率
            for i in range(self.n):
                self.attributes_probability[clazz].append(self.laplace_modified_condition_probability(data_belong_to_clazz[:, i]))

    def predict(self, sample):
        probability = []
        # 计算每个类的后验概率
        for clazz in self.classes_probability:
            ini = self.classes_probability[clazz]
            for i in range(self.n):
                ini *= self.attributes_classes_probability[clazz][i][sample[i]]
            probability.append({clazz: ini})
        return probability
