import numpy as np
import matplotlib.pyplot as plt
from QRFM import QRFM
import numpy as np
from load_data import read_1m
from QRPI import quantum_inspired, FM_recall
from QRFM import QRFM
import pandas as pd

# # 数据
# QRPI_time = [20.31, 33.97, 58.28, 83.34, 114.55, 152.13, 193.79, 240.57, 291.07]
# SVDpp_time = [14.32, 42.85, 117.04, 208.93, 362.50, 587.24, 852.16, 1145.48, 1330.69]
#
# # x轴数据（可以是任意连续的数值，这里使用从0到8的整数）
# # x = np.arange(len(QRPI_time))
# x = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#
#
# # 绘制原始数据
# plt.plot(x, QRPI_time, 'ro-', label='QRPI_time')
# plt.plot(x, SVDpp_time, 'bo-', label='SVDpp_time')
#
# # 多项式拟合
# degree = 2  # 多项式的次数
# fit_QRPI = np.polyfit(x, QRPI_time, degree)
# fit_SVDpp = np.polyfit(x, SVDpp_time, degree)
# fit_QRPI_time = np.polyval(fit_QRPI, x)
# fit_SVDpp_time = np.polyval(fit_SVDpp, x)
#
# # 绘制拟合曲线
# plt.plot(x, fit_QRPI_time, 'r--')
# plt.plot(x, fit_SVDpp_time, 'b--')
#
# # 添加图例和标签
# plt.legend()
# plt.xlabel('Index')
# plt.ylabel('Time (s)')
#
# # 显示图形
# plt.show


def QRPI_1m_test():
    res = QRFM(r=1)
    data2 = pd.read_csv("./QRPI_1m_result.csv")
    k = 10
    movies_test = np.load('A_movies_small_test.npy')
    x1 = 0
    x2 = 0
    m = 0
    for index, top_movies_ids in enumerate(res):
        i = index+1
        count = 0
        count2 = 0
        b = np.array(movies_test[i - 1])
        b_index = b.argsort()[::-1][0:k]
        accuracy = 0
        for index in np.array(top_movies_ids):
            if movies_test[i - 1][int(index) - 1] >= 4:
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
            m += 1
        x1 += count
        x2 += count2
    print('QRPI模型精度', x1/m)
    print('QRPI模型召回率', x2/m)
    return x1/m, x2/m


def FM_1m_test():
    res = QRFM(r=0)
    data2 = pd.read_csv("./QRPI_1m_result.csv")
    k = 10
    movies_test = np.load('A_movies_small_test.npy')
    x1 = 0
    x2 = 0
    m = 0
    for index, top_movies_ids in enumerate(res):
        i = index+1
        count = 0
        count2 = 0
        b = np.array(movies_test[i - 1])
        b_index = b.argsort()[::-1][0:k]
        accuracy = 0
        for index in np.array(top_movies_ids):
            if movies_test[i - 1][int(index) - 1] >= 4:
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
            m += 1
        x1 += count
        x2 += count2
    print('FM模型精度', x1/m)
    print('FM模型召回率', x2/m)


def QRFM_1m_test():
    res = QRFM()
    data2 = pd.read_csv("./QRPI_1m_result.csv")
    k = 10
    movies_test = np.load('A_movies_small_test.npy')
    x1 = 0
    x2 = 0
    m = 0
    for index, top_movies_ids in enumerate(res):
        i = index+1
        count = 0
        count2 = 0
        b = np.array(movies_test[i - 1])
        b_index = b.argsort()[::-1][0:k]
        accuracy = 0
        for index in np.array(top_movies_ids):
            if movies_test[i - 1][int(index) - 1] >= 4:
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
            m += 1
        x1 += count
        x2 += count2
    print('QRFM模型精度', x1/m)
    print('QRFM模型召回率', x2/m)
    return x1/m, x2/m


def QRPI_1m_test_avg_rr():
    res = QRFM(r=1)
    data2 = pd.read_csv("./QRPI_1m_result.csv")
    k = 10
    movies_test = np.load('A_movies_small_test.npy')
    mean_reciprocal_rank = 0  # 初始化平均倒数排名
    m = 0  # 初始化用户数
    for index, top_movies_ids in enumerate(res):
        i = index+1
        rank = 0  # 初始化排名
        for movie_id in top_movies_ids:
            rank += 1
            if movies_test[i - 1][int(movie_id) - 1] >= 4:  # 如果推荐的物品在测试集中被用户喜欢
                mean_reciprocal_rank += 1 / rank  # 更新平均倒数排名
                break  # 找到一个喜欢的物品就停止，因为只计算第一个喜欢的物品的排名
        m += 1  # 更新用户数
    mean_reciprocal_rank /= m  # 计算平均倒数排名
    print('QRPI平均倒数排名:', mean_reciprocal_rank)


def FM_1m_test_avg_rr():
    res = QRFM(r=0)
    data2 = pd.read_csv("./QRPI_1m_result.csv")
    k = 10
    movies_test = np.load('A_movies_small_test.npy')
    mean_reciprocal_rank = 0  # 初始化平均倒数排名
    m = 0  # 初始化用户数
    for index, top_movies_ids in enumerate(res):
        i = index+1
        rank = 0  # 初始化排名
        for movie_id in top_movies_ids:
            rank += 1
            if movies_test[i - 1][int(movie_id) - 1] >= 4:  # 如果推荐的物品在测试集中被用户喜欢
                mean_reciprocal_rank += 1 / rank  # 更新平均倒数排名
                break  # 找到一个喜欢的物品就停止，因为只计算第一个喜欢的物品的排名
        m += 1  # 更新用户数
    mean_reciprocal_rank /= m  # 计算平均倒数排名
    print('FM平均倒数排名:', mean_reciprocal_rank)



def QRFM_1m_test_avg_rr():
    res = QRFM()
    data2 = pd.read_csv("./QRPI_1m_result.csv")
    k = 10
    movies_test = np.load('A_movies_small_test.npy')
    mean_reciprocal_rank = 0  # 初始化平均倒数排名
    m = 0  # 初始化用户数
    for index, top_movies_ids in enumerate(res):
        i = index+1
        rank = 0  # 初始化排名
        for movie_id in top_movies_ids:
            rank += 1
            if movies_test[i - 1][int(movie_id) - 1] >= 4:  # 如果推荐的物品在测试集中被用户喜欢
                mean_reciprocal_rank += 1 / rank  # 更新平均倒数排名
                break  # 找到一个喜欢的物品就停止，因为只计算第一个喜欢的物品的排名
        m += 1  # 更新用户数
    mean_reciprocal_rank /= m  # 计算平均倒数排名
    print('QRFM平均倒数排名:', mean_reciprocal_rank)


if __name__ == '__main__':
    # QRPI_1m_test_avg_rr()
    # QRFM_1m_test_avg_rr()
    FM_1m_test_avg_rr()
    # QRFM_1m_test()
    # QRPI_1m_test()



