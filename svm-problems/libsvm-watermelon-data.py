# coding: utf-8
import pandas as pd
import os
from sklearn import svm
import numpy as np

data = pd.read_csv(os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__))), "test_data", "watermelon4.csv"), sep=" ")

# 密度和含糖率是 feature, 好瓜是 class
X = []
Y = []

for row in data.values:
    X.append(row[:-1])
    Y.append(row[-1])

# X 为 features, Y 为 class
X = np.array(X)
Y = np.array(Y)

# 用 libsvm 训练 SVM 模型
model = svm.libsvm.fit(X, Y)
