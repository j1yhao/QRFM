from split_train_test import split_data, mk_train_matrix2
from test import QRFM_1m_test, QRPI_1m_test
from QRPI import FM_recall
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def get_result():
    QRFM = []
    QRPI = []
    for i in range(0, 1):
        df, df_train, df_test, mu = split_data()
        A, B = mk_train_matrix2(df, df_train, mu)
        np.save("A_movies_small.npy", A)
        np.save("A_movies_small_test.npy", B)
        movies_train = np.load('A_movies_small.npy')
        movies_test = np.load('A_movies_small_test.npy')
        FM_recall(movies_train, movies_test)
        precisions, recall = QRFM_1m_test()
        QRFM.append(precisions)
        precisions, recall = QRPI_1m_test()
        QRPI.append(precisions)
    print(QRFM)
    print(QRPI)
    return QRFM, QRPI


if __name__ == '__main__':
    # QRFM_results, QRPI_results = get_result()
    QRFM_results = [0.776, 0.781, 0.7794, 0.7812, 0.7842, 0.7835, 0.7840, 0.7849, 0.7838, 0.7826]
    QRPI_results = [0.771, 0.777, 0.7769, 0.7761, 0.7791, 0.7785, 0.7788, 0.7788, 0.7782, 0.7785]
    # 执行t检验
    t_statistic, p_value = stats.ttest_ind(QRFM_results, QRPI_results)
    # 输出结果
    print("t统计量：", t_statistic)
    print("p值：", p_value)
    print("QRFM结果：", np.mean(QRFM_results))
    print("QRPI结果：", np.mean(QRPI_results))

    # 计算效应大小（Cohen's d）
    mean_diff = np.mean(QRFM_results) - np.mean(QRPI_results)
    pooled_std = np.sqrt((np.std(QRFM_results) ** 2 + np.std(QRPI_results) ** 2) / 2)
    effect_size = mean_diff / pooled_std
    print("效应大小（Cohen's d）：", effect_size)

    # 计算置信区间
    conf_interval = stats.t.interval(0.95, len(QRFM_results) - 1, loc=np.mean(QRFM_results) - np.mean(QRPI_results),
                                     scale=stats.sem(np.array(QRFM_results) - np.array(QRPI_results)))
    print("置信区间：", conf_interval)

    # 根据p值判断显著性
    alpha = 0.05
    if p_value < alpha:
        print("结果显著，拒绝原假设，QRFM和QRPI之间存在显著性差异。")
    else:
        print("结果不显著，无法拒绝原假设，QRFM和QRPI之间不存在显著性差异。")

    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.boxplot([QRPI_results, QRFM_results], labels=['QRPI precisions', 'QRFM precisions'])
    plt.title('Comparison of Prediction Precisions')
    plt.ylabel('Prediction Precisions')
    plt.grid(True)
    plt.show()
