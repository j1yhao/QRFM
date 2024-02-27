import numpy as np
from numpy import linalg as la
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
    coefficients = np.zeros((reps, rank))
    for i in range(reps):
        for l in range(rank):
            X = np.zeros(samples)
            for k in range(samples):
                sample_j = int(np.random.choice(n, 1, p=LS_prob_columns[user])[0])
                vl_j = 0
                for s in range(r):
                    vl_j += R[s, sample_j] * w[s, l]
                vl_j = vl_j / sigma[l]
                X[k] = vl_j * row_norms[user] / A[user, sample_j]
        coefficients[:, l] = np.mean(X)
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
    return x


if __name__ == '__main__':
    A = np.load('A_movies_small.npy')
    m, n = np.shape(A)
    print(m,n)
    user = 2
    r = 450
    c = 4500
    rank = 10
    Nsamples = 10000
    NcompX = 10
    X = quantum_inspired(A, user, r, c, rank, Nsamples, NcompX)
# LS_file = open('sample_C_File1.txt', "w+")
# LS_file.write(str(sampc))
# LS_file.close()
