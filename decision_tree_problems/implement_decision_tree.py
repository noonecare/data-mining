# coding: utf-8
import numpy as np

"""
不知道为什么用 sklearn 画出的决策树和西瓜书给出的答案不一样，所以我实现一下西瓜书中决策树的算法，画一下决策树的结果
"""


class Node:
    def __init__(self, attribute, samples):
        pass

    def decision(self):
        pass



def calc_ent(x):
    """
        calculate entropy
    """
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent


def calc_condition_ent(x, y):
    """
        calculate ent H(y|x)
    """

    # calc ent(y|x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent


def calc_ent_grap(x,y):
    """
        calculate ent grap
    """

    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent

    return ent_grap



class WaterMelonBookDecisionTree:
    """
    为了简单 DecisionTree 接受 二维的 Array 为 feature + class 最后一列为 class
    """

    def __init__(self, data):
        self.data = data
        # self.nodes 表示已经选出来的属性值
        self.tree = []
        # 每生成一个新的node, 给这个node 赋予 self.max_id + 1 的 id 并把这个 node 保存在 self.tree 的序列中
        self.max_id = -1


    def criterion(self, node: Node):


    def next_node(self, node: Node):
        """
        划分选择的算法，常见的有 gini 和 entroy 
        """
        # 如果 node 的 samples 个数为 0 范围 None
        if len(node.samples) == 0:
            return
        else:




