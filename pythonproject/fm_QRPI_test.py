import time

import numpy as np
from load_data import read_1m
from QRPI import quantum_inspired, FM_recall


def find_lcseque(s1, s2):
    # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    # d用来记录转移方向
    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 'ok'
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 'left'
            else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    # print(np.array(d))
    s = []
    while m[p1][p2]:  # 不为None时
        c = d[p1][p2]
        if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == 'left':  # 根据标记，向左找下一个
            p2 -= 1
        if c == 'up':  # 根据标记，向上找下一个
            p1 -= 1
    s.reverse()
    return s


def QRPI_1m_test_LCS():
    movies_train = np.load('A_movies_small.npy')
    p = 1000
    rank = 10
    result = quantum_inspired(movies_train, p, rank)
    movies_test = np.load('A_movies_small_test.npy')
    m, n = np.shape(result)
    count = 0
    for i in range(0, m):
        k = 10
        a = np.array(result[i])
        a_index = a.argsort()[::-1][0:n - 1]
        c_index = []
        for j in a_index:
            if movies_test[i][j] != 0:
                c_index.append(j)
            if len(c_index) == k:
                break
        b = np.array(movies_test[i])
        b_index = b.argsort()[::-1][0:k]
        index = []
        for j in range(0, len(b_index)):
            if b[b_index[j]] != 0:
                index.append(j)
        b_index = b_index[index]
        # s = find_lcseque(c_index, b_index)
        s = find_lcseque(np.sort(c_index), np.sort(b_index))
        if len(b_index) != 0:
            lcs = len(s) / len(b_index)
            count += lcs
        else:
            m -= 1
    print('QRPI算法LCS评分： ', count/m)
    return count / m



def QRPI_1m_test_Accuracy():
    movies_train = np.load('A_movies_small.npy')
    p = 1000
    rank = 10
    result = quantum_inspired(movies_train, p, rank)
    movies_test = np.load('A_movies_small_test.npy')
    m, n = np.shape(result)
    count = 0
    for i in range(0, m):
        k = 10
        a = np.array(result[i])
        a_index = a.argsort()[::-1][0:n - 1]
        c_index = []
        for j in a_index:
            if movies_test[i][j] != 0:
                c_index.append(j)
            if len(c_index) == 10:
                break
        b = np.array(movies_test[i])
        b_index = b.argsort()[::-1][0:k]
        index = []
        for j in range(0, len(b_index)):
            if b[b_index[j]] != 0 and b[b_index[j]] >= 4:
                index.append(j)
        b_index = b_index[index]
        accuracy = 0
        for index in c_index:
            if movies_test[i][index] >= 4:
                accuracy += 1
        if len(b_index) != 0:
            lcs = accuracy / k
            count += lcs
        else:
            m -= 1
    print('QRPI算法精确度： ', count/m)
    return count / m


if __name__ == '__main__':
    # a = QRPI_1m_test_LCS()
    b = QRPI_1m_test_Accuracy()
    # print("f1-measure:", (2*a*b)/(a+b))

