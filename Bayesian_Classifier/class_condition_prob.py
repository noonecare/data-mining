# coding: utf-8
import numpy as np
import pandas as pd


def calculate_condition_probability(x: np.ndarray):
    """
    计算分布
    :param x:一系列的数 
    :return: x 中每个 value 出现的概率
    """
    distinct_values = np.unique(x)
    length = len(x)
    frequency = [(value, len(x[x == value]) / length) for value in distinct_values]

    return frequency


if __name__ == '__main__':
    # 读取西瓜数据集3
    data = pd.read_csv(r"../test_data/watermelon1.csv")
    # 前三个属性每个属性的类条件概率为
    # 类为是的条件概率
    good_melon = data[data["好瓜"] == "是"]
    print(calculate_condition_probability(good_melon["色泽"].values))
    print(calculate_condition_probability(good_melon["根蒂"].values))
    print(calculate_condition_probability(good_melon["敲声"].values))
    # 类为否的条件概率
    not_good_melon = data[data["好瓜"] == "否"]
    print(calculate_condition_probability(not_good_melon["色泽"].values))
    print(calculate_condition_probability(not_good_melon["根蒂"].values))
    print(calculate_condition_probability(not_good_melon["敲声"].values))
