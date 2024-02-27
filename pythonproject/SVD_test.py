from SVDpp import SVDPP, getMLData
import numpy as np


if __name__ == '__main__':
    # train_data, test_data = getMLData2(1)
    # a = SVDPP(train_data, 1)
    # # a.train()
    # a.test2(test_data)
    result = []
    result1 = []
    x = []
    movies_train = np.load('A_movies_small.npy')
    movies_test = np.load('A_movies_small_test.npy')
    # m, n = np.shape(movies_train)
    # m1 = int(m * 0.1)
    # n1 = int(n * 0.1)
    # movies_train1 = movies_train[:m1, :n1]
    # movies_test1 = movies_test[:m1, :n1]
    a = SVDPP(getMLData(movies_train))
    a.train(steps=10)
    # a.test2(getMLData(movies_test))
    a.test_avg_rr(getMLData(movies_test))
