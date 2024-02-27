import math

import pandas as pd
import numpy as np
import random


def split_data(split_ratio=1):
    # filename = "./data/ml-100k/u.data"
    filename = "./data/ml-1m/ratings.dat"
    rname = ['user_id', 'movie_id', 'rating', 'timestamp']
    df = pd.read_table(filename, sep='::', header=None, names=rname, engine='python')
    df.drop(['timestamp'], axis=1)
    df = df.sample(frac=1)
    # split_ratio = 1
    cut_off = int(len(df) * split_ratio)
    # 使用训练数据生成矩阵
    df_train = df.iloc[:cut_off]
    df_test = df.iloc[cut_off:]
    mu = sum(df_train.rating.values) / len(df_train.rating.values)
    return df, df_train, df_test, mu


def split_data2(split_ratio):
    print("split_data")
    filename = "./data/ml-100k/u.data"
    # filename = "./data/ml-1m/ratings.dat"
    rname = ['user_id', 'movie_id', 'rating', 'timestamp']
    df = pd.read_table(filename, sep='\t', header=None, names=rname, engine='python')
    df.drop(['timestamp'], axis=1)
    df = df.sample(frac=1)
    cut_off = int(len(df) * split_ratio)
    # 使用训练数据生成矩阵
    df_train = df.iloc[:cut_off]
    df_test = df.iloc[cut_off:]
    mu = sum(df_train.rating.values) / len(df_train.rating.values)
    return df, df_train, df_test, mu


def mk_train_matrix(df, df_train, mu):
    print("mk_train_matrix")
    N = df.user_id.max() + 1
    M = df.movie_id.max() + 1
    A = np.zeros([N, M])
    for i in range(N):
        for j in range(M):
            A[i, j] = mu
    for row in df_train.iterrows():
        A[int(row[1]['user_id']),int(row[1]['movie_id'])] = float(row[1]['rating'])
    return np.mat(A)


def mk_train_matrix2(df, df_train, mu):
    print("mk_train_matrix")
    N = df.user_id.max()
    M = df.movie_id.max()
    train_data = np.zeros([N, M])
    test_data = np.zeros([N, M])
    for i in range(N):
        for j in range(M):
            train_data[i, j] = 0
            test_data[i, j] = 0
    for row in df_train.iterrows():
        c = random.randint(1,2)
        if c == 1:
            train_data[int(row[1]['user_id'])-1,int(row[1]['movie_id'])-1] = float(row[1]['rating'])
        else:
            test_data[int(row[1]['user_id']-1),int(row[1]['movie_id'])-1] = float(row[1]['rating'])
    return np.mat(train_data), np.mat(test_data)
    # return np.mat(train_data)[0:int(N*0.5),0:int(M*0.5)], np.mat(test_data)[0:int(N*0.5),0:int(M*0.5)]


def res_evaluation(Ak,df_test):
    print("res_evaluation")
    RMSE = 0
    count = 0
    for row in df_test.iterrows():
        # print(f"Ak : {Ak[int(row[1]['user_id']), int(row[1]['movie_id'])]},A : {row[1]['rating']}")
        count += 1
        RMSE += (Ak[int(row[1]['user_id']), int(row[1]['movie_id'])] - row[1]['rating']) ** 2
    RMSE = math.sqrt(RMSE / count)
    print(RMSE)


if __name__ == '__main__':
    # df, df_train, df_test, mu = split_data()
    df, df_train, df_test, mu = split_data()
    A, B = mk_train_matrix2(df, df_train, mu)
    np.save("A_movies_small.npy", A)
    np.save("A_movies_small_test.npy", B)
    # A = mk_train_matrix(df, df_train, mu)
    # np.save("A_movies_small.npy", A)








