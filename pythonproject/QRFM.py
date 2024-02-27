import xlearn as xl
import pandas as pd
import numpy as np
from QRPI import FM_recall
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import time


fm_model = xl.create_fm()


fm_model.setTrain("./FM_model_train.csv")  # 训练数据路径
fm_model.setValidate("./FM_model_test.csv")
fm_model.setSigmoid()  # 使用sigmoid函数作为输出层激活函数

# 设置训练参数
param = {
    'task': 'binary',
    'lr': 0.2,
    'lambda': 0.0000005,
    'k': 12,
    'epoch': 50,
    'metric': 'acc',
    'nthread':4,
}

# 训练FM模型
# fm_model.fit(param, "fm_model.out")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def QRFM(k=10, r=0.75):
    # movies_train = np.load('A_movies_small.npy')
    # movies_test = np.load('A_movies_small_test.npy')
    # FM_recall(movies_train, movies_test)
    data2 = pd.read_csv("./QRPI_1m_result.csv")
    data2_rate = data2
    data2 = data2.drop("rate", axis=1)
    res = []
    for i in range(1, 6000):
        user_data = data2[data2['user_id'] == i]
        user_rate = data2_rate[data2_rate['user_id'] == i]
        user_data1 = user_data.drop("user_id", axis=1)
        user_data2 = user_data1.drop("movie_id", axis=1)
        user_test = xl.DMatrix(user_data2)
        fm_model.setTest(user_test)
        result = fm_model.predict("fm_model.out")
        QRPI_rate = sigmoid(np.array(user_rate['rate']))
        result = ((1-r) * result) + (r * QRPI_rate)
        top_movies_indices = result.argsort()[::-1][0:k]
        top_movies_ids = user_data.iloc[top_movies_indices]["movie_id"].tolist()
        res.append(top_movies_ids)
    return res


if __name__ == '__main__':
   QRFM()





