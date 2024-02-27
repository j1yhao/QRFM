import time

import numpy as np

movies_train = np.load('A_movies_small.npy')
movies_test = np.load('A_movies_small_test.npy')


def user_based_cf_recommendation(train_data, test_data, k=10, N=10):
    """
    基于用户的协同过滤推荐算法，计算topN推荐的准确率和召回率
    :param train_data: 训练集，用户物品评分矩阵
    :param test_data: 测试集，用户物品评分矩阵
    :param k: 考虑的相似用户数量
    :param N: 推荐的物品数量
    :return: 准确率和召回率
    """
    precision = 0
    recall = 0
    total_users = 0

    for user_id in range(len(test_data)):
        # 获取用户在训练集中的评分记录
        user_ratings_train = train_data[user_id]

        # 找出用户在测试集中实际评分的物品
        actual_ratings = test_data[user_id]
        actual_items = [item_id for item_id, rating in enumerate(actual_ratings) if rating >= 4]

        # 如果用户在训练集中没有评分记录，则跳过
        if not user_ratings_train.any():
            continue

        # 计算用户与其他用户的相似度
        similarities = []
        for other_id, other_ratings in enumerate(train_data):
            if other_id != user_id and other_ratings.any():
                similarity = cosine_similarity(user_ratings_train, other_ratings)
                similarities.append((other_id, similarity))

        # 根据相似度排序，取top k相似用户
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_users = similarities[:k]

        # 统计topN推荐的物品
        recommended_items = set()
        for other_id, _ in similar_users:
            other_ratings = train_data[other_id]
            for item_id, rating in enumerate(other_ratings):
                if rating >= 4 and item_id not in user_ratings_train:
                    recommended_items.add(item_id)
                    if len(recommended_items) == N:  # 修正推荐的物品数量为N
                        break
            if len(recommended_items) == N:
                break
        if len(actual_items) != 0:
            # 计算准确率和召回率
            num_relevant_items = len(set(actual_items) & recommended_items)
            precision += num_relevant_items / len(recommended_items)
            recall += num_relevant_items / len(actual_items)
            total_users += 1

    # 计算平均准确率和召回率
    precision /= total_users
    recall /= total_users

    return precision, recall


# 计算余弦相似度
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


# 示例使用
t1 = time.time()
precision, recall = user_based_cf_recommendation(movies_train, movies_test, k=5, N=10)
t2 = time.time()
print("平均准确率:", precision)
print("平均召回率:", recall)
print("运行时间：", t2 - t1)

