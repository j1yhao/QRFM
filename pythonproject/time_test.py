import numpy as np
from QRPI import FM_recall
import xlearn as xl
import pandas as pd
import time
import matplotlib.pyplot as plt
from SVDpp import getMLData, SVDPP


fm_model = xl.create_fm()


fm_model.setTrain("./FM_model_train.csv")  # 训练数据路径
fm_model.setValidate("./FM_model_test.csv")
fm_model.setSigmoid()  # 使用sigmoid函数作为输出层激活函数


if __name__ == '__main__':
    movies_train = np.load('A_movies_small.npy')
    movies_test = np.load('A_movies_small_test.npy')
    m, n = np.shape(movies_train)
    QRPI_time = []
    SVDpp_time = [14.32, 42.85, 117.04, 208.93, 362.50, 587.24, 852.16, 1145.48, 1330.69]
    x = []
    for i in range(2, 11):
        split_ratio = i / 10
        m1 = int(m * split_ratio)
        n1 = int(n * split_ratio)
        movies_train1 = movies_train[:m1, :n1]
        movies_test1 = movies_test[:m1, :n1]
        # result.append(FM_recall(movies_train1, movies_test1))
        # a = SVDPP(getMLData(movies_train1))
        # result1.append(a.train())
        # x.append(split_ratio)

        time1 = FM_recall(movies_train1, movies_test1)
        t1 = time.time()
        data2 = pd.read_csv("./QRPI_1m_result.csv")
        k = 10
        data2_rate = data2
        data2 = data2.drop("rate", axis=1)
        for i in range(1, 6000):
            user_data = data2[data2['user_id'] == i]
            user_rate = data2_rate[data2_rate['user_id'] == i]
            user_data1 = user_data.drop("user_id", axis=1)
            user_data2 = user_data1.drop("movie_id", axis=1)
            user_test = xl.DMatrix(user_data2)
            fm_model.setTest(user_test)
        t2 = time.time()
        QRPI_time.append(time1+t2-t1)
        print(time1+t2-t1)
        # a = SVDPP(getMLData(movies_train1))
        # SVDpp_time.append(a.train())
        x.append(split_ratio)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams["font.family"] = "Times New Roman"
    plt.plot(x, SVDpp_time, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='SVD++')
    plt.plot(x, QRPI_time, 'ro-', color='red', alpha=0.8, linewidth=1, label='QRFM')
    plt.legend(loc="best")
    plt.xlabel('Data Size')
    plt.ylabel('Running Time')
    plt.savefig('plot.png')
    plt.show()