import numpy as np
from numpy import linalg  as la
import time
from multiprocessing import Pool

def ls_probs(m, n, A):
    tic = time.time()
    row_norms = np.zeros(m)
    sum = 0
    for i in range(m):
        row_norms[i] = la.norm(A[i, :]) ** 2
        sum += row_norms[i]
    A_Frobenius = np.sqrt(sum)

    LS_prob_rows = np.zeros(m)
    for i in range(m):
        LS_prob_rows[i] = row_norms[i] / A_Frobenius ** 2

    LS_prob_columns = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            LS_prob_columns[i][j] = A[i, j] ** 2 / row_norms[i]
    toc = time.time()

    rt_LS_probs = toc - tic
    print("2范数采样时间 : " + str(rt_LS_probs))
    return row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius


def sample_C(A, m, n, r, c, row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius):
    # 采样i和j
    tic = time.time()
    rows = np.random.choice(m, r, replace=True, p=LS_prob_rows)
    columns = np.zeros(c, dtype=int)
    for j in range(c):
        i = np.random.choice(rows, replace=True)
        columns[j] = np.random.choice(n, 1, p=LS_prob_columns[i])

    # 生成行标准化过的R
    R = np.zeros((r, n))
    for i in range(r):
        R[i, :] = A[rows[i], :] * A_Frobenius / (np.sqrt(r) * np.sqrt(row_norms[rows[i]]))

    # 生成双标准化过的C
    C = np.zeros((r, c))
    column_norms = np.zeros(c)
    for i in range(r):
        for j in range(c):
            C[i, j] = A[rows[i], columns[j]] * A_Frobenius / (np.sqrt(r) * np.sqrt(row_norms[rows[i]]))
            column_norms[j] += C[i, j] ** 2

    for j in range(c):
        C[:, j] = C[:, j] * A_Frobenius / (np.sqrt(c) * np.sqrt(column_norms[j]))

    w, sigma, vh = la.svd(C, full_matrices=False)
    toc = time.time()
    rt_sampling_C = toc - tic
    print("生成R、C并完成C的svd的时间 ：" + str(rt_sampling_C))
    return w, sigma, vh, rows, R, LS_prob_columns


def sample_me_rsys(A, user, n, samples, rank, r, w, rows, sigma, row_norms, LS_prob_columns, A_Frobenius, R):
    tic = time.time()
    reps = 10
    coefficients = np.zeros((reps, rank))
    for rep in range(reps):
        for l in range(rank):
            X = np.zeros(samples)
            for k in range(samples):
                sample_j = int(np.random.choice(n, 1, p=LS_prob_columns[user])[0])
                vl_j = 0
                for s in range(r):
                    vl_j += R[s, sample_j] * w[s, l]
                vl_j = vl_j / sigma[l]
                # tic2 = time.time()
                # vl_j = 0
                # for s in range(r):
                # 	vl_j += A[rows[s], sample_j] * w[s, l] / (np.sqrt(row_norms[rows[s]]))
                # vl_j = vl_j * A_Frobenius / (np.sqrt(r) * sigma[l])
                # toc2 = time.time()
                # t2 += toc2 - tic2
                X[k] = vl_j * row_norms[user] / A[user, sample_j]

            coefficients[rep, l] = np.mean(X)
    lambdas = np.zeros(rank)
    for l in range(rank):
        lambdas[l] = np.median(coefficients[:, l])
    toc = time.time()
    rt_sample_me_rsys = toc - tic
    print("采样估计λ的时间 ： " + str(rt_sample_me_rsys))
    return lambdas


def sample_me_rsys(A, user, n, samples, rank, r, w, rows, sigma, row_norms, LS_prob_columns, A_Frobenius, R):
    tic = time.time()
    reps = 10
    coefficients = np.zeros(reps)
    PM = [rank, samples, r, n, LS_prob_columns, R, w, sigma, row_norms, A, user]
    PMs = []
    for i in range(reps):
        PMs.append(PM)
    with Pool(reps) as p:
        coefficients = np.array(p.map(sub_sample_me_rsys, PMs))
    lambdas = np.zeros(rank)
    for l in range(rank):
        lambdas[l] = np.median(coefficients[:, l])
    toc = time.time()
    rt_sample_me_rsys = toc - tic
    print("采样估计λ的时间 ： " + str(rt_sample_me_rsys))
    return lambdas


def sub_sample_me_rsys(PM):
    rank, samples, r, n, LS_prob_columns, R, w, sigma, row_norms, A, user = PM[:]
    coefficient = np.zeros(rank)
    for l in range(rank):
        X = np.zeros(samples)
        for k in range(samples):
            sample_j = int(np.random.choice(n, 1, p=LS_prob_columns[user])[0])
            vl_j = 0
            for s in range(r):
                vl_j += R[s, sample_j] * w[s, l]
            vl_j = vl_j / sigma[l]
            X[k] = vl_j * row_norms[user] / A[user, sample_j]
        coefficient[l] = np.mean(X)

    return coefficient


def sample_from_x(A, r, n, LS_prob_columns, w_vector, w_norm, R):
    keepGoing = True
    counter = 0
    out_j = 0
    out_dot = 0
    while keepGoing:
        counter += 1
        i_sample = np.random.choice(r)
        j_sample = np.random.choice(n, 1, p=LS_prob_columns[i_sample])[0]
        R_j = np.zeros(n)
        R_j = R[:, j_sample]
        R_j_norm = la.norm(R_j)
        RW_dot = np.dot(R_j, w_vector)
        if R_j_norm == 0 or w_norm == 0:
            coin = 0
        else:
            prob = (RW_dot / (R_j_norm * w_norm)) ** 2
            coin = np.random.binomial(1, prob)
        if coin == 1:
            out_j = j_sample
            out_dot = RW_dot
            keepGoing = False

    return int(out_j), out_dot


def quantum_inspired(A, user, r, c, rank, Nsamples, NcompX):
    t1 = time.time()
    m, n = np.shape(A)
    LS = ls_probs(m, n, A)
    row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius = LS[0:4]
    svd_C = sample_C(A, m, n, r, c, row_norms, LS_prob_rows, LS_prob_columns, A_Frobenius)
    w, sigma, vh, rows, R = svd_C[0:5]
    lambdas = sample_me_rsys(A, user, n, Nsamples, rank, r, w, rows, sigma, row_norms, LS_prob_columns, A_Frobenius, R)
    w_vector = np.zeros(r)
    for l in range(rank):
        w_vector[:] += (lambdas[l] / sigma[l]) * w[:, l]
    w_norm = la.norm(w_vector)
    j_index = np.zeros(NcompX)
    x = np.zeros(NcompX)
    tic = time.time()
    for i in range(NcompX):
        j_index[i], x[i] = sample_from_x(A, r, n, LS_prob_columns, w_vector, w_norm, R)
    toc = time.time()
    rt_sample_from_x = toc - tic
    print("采样生成解向量的时间 ：" + str(rt_sample_from_x))
    t2 = time.time()
    print("总时间 : " + str(t2 - t1))
    return x, j_index


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


if __name__ == '__main__':
    A = np.load('A_movies_small.npy')
    r = 450
    c = 4500
    rank = 10
    Nsamples = 10000
    NcompX = 1500
    movies_test = np.load('A_movies_small_test.npy')
    count = 0
    count2 = 0
    for user in range(300):
        X, j_index = quantum_inspired(A, user, r, c, rank, Nsamples, NcompX)
        k = 10
        a_index = X.argsort()[::-1][0:NcompX]
        a2_index = []
        flag = 0
        for j in a_index:
            if j_index[j] not in a2_index and movies_test[user][int(j_index[j])] != 0:
                a2_index.append(j_index[j])
                flag += 1
            if flag == k:
                break
        b = np.array(movies_test[user])
        b_index = b.argsort()[::-1][0:k]
        index = []
        for j in range(0, len(b_index)):
            if b[b_index[j]] != 0:
                index.append(j)
        b_index = b_index[index]
        s = find_lcseque(np.sort(a2_index), np.sort(b_index))
        count += len(s) / len(b_index)
        accuracy = 0
        for j in a2_index:
            if movies_test[user][int(j)] >= 4:
                accuracy += 1
        count2 += accuracy / len(b_index)
        print(len(s)/k)
        print(accuracy / len(b_index))
    print(count/300)
# LS_file = open('sample_C_File1.txt', "w+")
# LS_file.write(str(sampc))
# LS_file.close()
