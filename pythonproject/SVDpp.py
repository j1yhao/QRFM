import numpy as np
from numpy import linalg as la
import time
from split_train_test import split_data2, mk_train_matrix2, split_data
import matplotlib.pyplot as plt
import random
from QRPI import FM_recall


class SVDPP:
    def __init__(self, mat, K=20):
        self.mat = np.array(mat)
        self.K = K
        self.bi = {}
        self.bu = {}
        self.qi = {}
        self.pu = {}
        self.avg = np.mean(self.mat[:, 2])
        self.y = {}
        self.u_dict = {}
        for i in range(self.mat.shape[0]):
            uid = self.mat[i, 0]
            iid = self.mat[i, 1]
            self.u_dict.setdefault(uid, [])
            self.u_dict[uid].append(iid)
            self.bi.setdefault(iid, 0)
            self.bu.setdefault(uid, 0)
            self.qi.setdefault(iid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))
            self.pu.setdefault(uid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))
            self.y.setdefault(iid, np.zeros((self.K, 1)) + .1)

    def predict(self, uid, iid):  # 预测评分的函数
        # setdefault的作用是当该用户或者物品未出现过时，新建它的bi,bu,qi,pu及用户评价过的物品u_dict，并设置初始值为0
        self.bi.setdefault(iid, 0)
        self.bu.setdefault(uid, 0)
        self.qi.setdefault(iid, np.zeros((self.K, 1)))
        self.pu.setdefault(uid, np.zeros((self.K, 1)))
        self.y.setdefault(uid, np.zeros((self.K, 1)))
        self.u_dict.setdefault(uid, [])
        u_impl_prf, sqrt_Nu = self.getY(uid, iid)
        rating = self.avg + self.bi[iid] + self.bu[uid] + np.sum(self.qi[iid] * (self.pu[uid] + u_impl_prf))  # 预测评分公式
        # 由于评分范围在1到5，所以当分数大于5或小于1时，返回5,1.
        if rating > 5:
            rating = 5
        if rating < 1:
            rating = 1
        return rating

    # 计算sqrt_Nu和∑yj
    def getY(self, uid, iid):
        Nu = self.u_dict[uid]
        I_Nu = len(Nu)
        sqrt_Nu = np.sqrt(I_Nu)
        y_u = np.zeros((self.K, 1))
        if I_Nu == 0:
            u_impl_prf = y_u
        else:
            for i in Nu:
                y_u += self.y[i]
            u_impl_prf = y_u / sqrt_Nu

        return u_impl_prf, sqrt_Nu

    def train(self, steps=3, gamma=0.04, Lambda=0.15):  # 训练函数，step为迭代次数。
        t1 = time.time()
        print('train data size', self.mat.shape)
        for step in range(steps):
            print('step', step + 1, 'is running')
            KK = np.random.permutation(self.mat.shape[0])  # 随机梯度下降算法，kk为对矩阵进行随机洗牌
            rmse = 0.0
            for i in range(self.mat.shape[0]):
                j = KK[i]
                uid = self.mat[j, 0]
                iid = self.mat[j, 1]
                rating = self.mat[j, 2]
                predict = self.predict(uid, iid)
                u_impl_prf, sqrt_Nu = self.getY(uid, iid)
                eui = rating - predict
                rmse += eui ** 2
                self.bu[uid] += gamma * (eui - Lambda * self.bu[uid])
                self.bi[iid] += gamma * (eui - Lambda * self.bi[iid])
                self.pu[uid] += gamma * (eui * self.qi[iid] - Lambda * self.pu[uid])
                self.qi[iid] += gamma * (eui * (self.pu[uid] + u_impl_prf) - Lambda * self.qi[iid])
                for j in self.u_dict[uid]:
                    self.y[j] += gamma * (eui * self.qi[j] / sqrt_Nu - Lambda * self.y[j])

            gamma = 0.93 * gamma
            print('rmse is', np.sqrt(rmse / self.mat.shape[0]))
        t2 = time.time()
        print("运行时间:", t2-t1)
        return t2-t1

    def test(self, test_data):  # gamma以0.93的学习率递减
        test_data = np.array(test_data)
        print('test data size', test_data.shape)
        rmse = 0.0
        for i in range(test_data.shape[0]):
            uid = test_data[i, 0]
            iid = test_data[i, 1]
            rating = test_data[i, 2]
            eui = rating - self.predict(uid, iid)
            rmse += eui ** 2
        print('rmse of test data is', np.sqrt(rmse / test_data.shape[0]))

    def test2(self, test_data):
        test_data = np.array(test_data)
        m = 6040
        n = 3952
        test_data2 = np.zeros([m, n])
        result_data = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                test_data2[i, j] = 0
                result_data[i, j] = 0
        for i in range(len(test_data)):
            print(test_data[i, 0], test_data[i, 1], test_data[i, 2])
            test_data2[int(test_data[i, 0]), int(test_data[i, 1])] = int(test_data[i, 2])
            result_data[int(test_data[i, 0]), int(test_data[i, 1])] = self.predict(int(test_data[i, 0]),
                                                                                   int(test_data[i, 1]))
        LCS = 0
        count = 0
        acc = 0
        for i in range(m):
            k = 10
            a = np.array(result_data[i])
            a_index = a.argsort()[::-1][0:k]
            b = np.array(test_data2[i])
            b_index = b.argsort()[::-1][0:k]
            index = []
            for j in range(0, len(b_index)):
                if b[b_index[j]] != 0:
                    index.append(j)
            b_index = b_index[index]
            accuracy = 0
            for j in a_index:
                if test_data2[i][j] >= 4:
                    accuracy += 1
            common_elements = set(a_index) & set(b_index)
            if len(b_index) != 0:
                acc += accuracy / len(b_index)
                LCS += len(common_elements) / len(b_index)
                count += 1
        print(LCS/count)
        print(acc/count)
        return LCS/count

    def test_avg_rr(self, test_data):
        test_data = np.array(test_data)
        m = 6040
        n = 3952
        test_data2 = np.zeros([m, n])
        result_data = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                test_data2[i, j] = 0
                result_data[i, j] = 0
        for i in range(len(test_data)):
            print(test_data[i, 0], test_data[i, 1], test_data[i, 2])
            test_data2[int(test_data[i, 0]), int(test_data[i, 1])] = int(test_data[i, 2])
            result_data[int(test_data[i, 0]), int(test_data[i, 1])] = self.predict(int(test_data[i, 0]),
                                                                                   int(test_data[i, 1]))
        total_rr = 0
        num_queries = 0
        for i in range(m):
            k = 10
            a = np.array(result_data[i])
            a_index = a.argsort()[::-1][0:k]
            b = np.array(test_data2[i])
            b_index = b.argsort()[::-1][0:k]
            index = []
            for j in range(0, len(b_index)):
                if b[b_index[j]] != 0:
                    index.append(j)
            b_index = b_index[index]
            accuracy = 0
            for j in a_index:
                if test_data2[i][j] >= 4:
                    accuracy += 1
            common_elements = set(a_index) & set(b_index)
            if len(b_index) != 0:
                rr = len(common_elements) / len(b_index)
                total_rr += rr
                num_queries += 1
        avg_rr = total_rr / num_queries if num_queries != 0 else 0
        print('平均倒数排名：', avg_rr)
        return avg_rr


def getMLData(data):
    m,n = np.shape(data)
    data2 = []
    for i in range(m):
        for j in range(n):
            if data[i][j] != 0:
                data2.append([i,j,data[i][j]])
    return data2


def getMLData2(split_ratio):  # 获取训练集和测试集的函数
    import re
    f = open("./data/ml-1m/ratings.dat", 'r')
    lines = f.readlines()
    f.close()
    data = []
    data2 = []
    count = 0
    for line in lines:
        list = re.split('::|\n', line)
        if int(list[2]) != 0:
            c = random.randint(1, 2)
            if c == 1:
                data.append([int(i) for i in list[:3]])
            else:
                data2.append([int(i) for i in list[:3]])
        count += 1
        if count == int(len(lines)*split_ratio):
            break
    train_data = data
    test_data = data2
    return train_data, test_data


def svd_test():
    result = []
    for i in range(2, 11):
        split_ratio = i / 10
        train_data, test_data = getMLData2(split_ratio)
        a = SVDPP(train_data, 3)
        time = a.train()
        result.append(time)
    return result


if __name__ == '__main__':
    # train_data, test_data = getMLData2(1)
    # a = SVDPP(train_data, 1)
    # # a.train()
    # a.test2(test_data)
    QRPI_time = [20.31, 33.97, 58.28, 83.34, 114.55, 152.13, 193.79, 240.57, 291.07]
    SVDpp_time = [14.32, 42.85, 117.04, 208.93, 362.50, 587.24, 852.16, 1145.48, 1330.69]
    result = []
    result1 = []
    x = []
    movies_train = np.load('A_movies_small.npy')
    movies_test = np.load('A_movies_small_test.npy')
    m, n = np.shape(movies_train)
    for i in range(2, 11):
        split_ratio = i / 10
        m1 = int(m * split_ratio)
        n1 = int(n * split_ratio)
        movies_train1 = movies_train[:m1, :n1]
        movies_test1 = movies_test[:m1, :n1]
        result.append(FM_recall(movies_train1, movies_test1))
        a = SVDPP(getMLData(movies_train1))
        result1.append(a.train())
        x.append(split_ratio)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(x, result, 'ro-', color='#4169E1', alpha=0.8, linewidth=1, label='QRPI time')
    plt.plot(x, result1, 'ro-', color='red', alpha=0.8, linewidth=1, label='SVD++ time')
    plt.legend(loc="best")
    plt.xlabel('Data Size')
    plt.ylabel('Running Time')
    plt.savefig('plot.png')
    plt.show()
