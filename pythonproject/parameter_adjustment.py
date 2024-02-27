import xlearn as xl
import pandas as pd
import numpy as np
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


# fm_model.fit(param, "fm_model.out")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def FM_predict():
    data2 = pd.read_csv("./QRPI_1m_result.csv")
    movies_test = np.load('A_movies_small_test.npy')
    data2 = data2.drop("rate", axis=1)
    m, n = np.shape(movies_test)
    for i in range(m):
        user_data = data2[data2['user_id'] == i]
        user_data = user_data.drop("user_id", axis=1)
        user_data = user_data.drop("movie_id", axis=1)
        user_test = xl.DMatrix(user_data)
        fm_model.setTest(user_test)
        # result = fm_model.predict("fm_model.out")


def test():
    data2 = pd.read_csv("./QRPI_1m_result.csv")
    k = 10
    movies_test = np.load('A_movies_small_test.npy')
    data2_rate = data2
    data2 = data2.drop("rate", axis=1)
    x1 = []
    x2 = []
    y = []
    QRPI = []
    QIRS = []
    m = 0
    for r in range(0, 20):
        x1.append(0)
        x2.append(0)
        y.append(r/20)
        # QIRS.append(0.73)
        # QRPI.append(0.774)
        # QIRS.append(0.36)
        # QRPI.append(0.40)
    for i in range(1, 6000):
        user_data = data2[data2['user_id'] == i]
        user_rate = data2_rate[data2_rate['user_id'] == i]
        user_data1 = user_data.drop("user_id", axis=1)
        user_data2 = user_data1.drop("movie_id", axis=1)
        user_test = xl.DMatrix(user_data2)
        fm_model.setTest(user_test)
        result = fm_model.predict("fm_model.out")
        QRPI_rate = sigmoid(np.array(user_rate['rate']))
        for r in range(0, 20):
            count = 0
            count2 = 0
            r1 = r / 20
            result = ((1-r1)*result) + (r1*QRPI_rate)
            top_movies_indices = result.argsort()[::-1][0:k]
            top_movies_ids = user_data.iloc[top_movies_indices]["movie_id"].tolist()
            b = np.array(movies_test[i-1])
            b_index = b.argsort()[::-1][0:k]
            accuracy = 0
            for index in np.array(top_movies_ids):
                if movies_test[i-1][int(index)-1] >= 4:
                    accuracy += 1
            index = []
            for j in range(0, len(b_index)):
                if b[b_index[j]] != 0:
                    index.append(j)
            b_index = b_index[index]
            b_index = [x + 1 for x in b_index]
            common_elements = set(top_movies_ids) & set(b_index)
            if len(b_index) != 0:
                lcs = accuracy / len(b_index)
                count += lcs
                count2 += len(common_elements) / len(b_index)
                if r == 0:
                    m += 1
            x1[r] += count
            x2[r] += count2
    x11 = [x / m for x in x1]
    x22 = [x / m for x in x2]
    x3 = [2 * i * j for i,j in zip(x11, x22)]
    x4 = [i+j for i,j in zip(x11, x22)]
    x5 = [i/j for i,j in zip(x3, x4)]
    for i in range(20):
        QIRS.append(x11[0])
        QRPI.append(x11[-1])
    plt.rcParams["font.family"] = "Times New Roman"
    plt.plot(y, x11, 'ro-', color='red', alpha=0.8, linewidth=1, label='QRFM')
    plt.plot(y, QRPI, 'r--', color='green', alpha=0.8, linewidth=1, label='QRPI')
    plt.plot(y, QIRS, 'r--', color='blue', alpha=0.8, linewidth=1, label='FM')

    plt.legend(loc='best')
    plt.xlabel('QRPI rate weight')

    plt.ylabel('recall')
    print('混合模型精度', x11)
    # print('混合模型召回率', x22)
    plt.show()

    # return count/m, count2/m


if __name__ == '__main__':
    test()




