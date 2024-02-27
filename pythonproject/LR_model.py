import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


# 准备训练集和测试集数据
train_data = [
    [1, 1, 5],
    [1, 2, 4],
    [2, 1, 3],
    [2, 3, 2],
    [3, 2, 4],
    [3, 3, 5]
]

test_data = [
    [1, 3],
    [2, 2],
    [3, 1]
]

# 提取特征和标签
X_train = np.array([data[:2] for data in train_data])
y_train = np.array([data[2] for data in train_data])

X_test = np.array([data for data in test_data])

# 特征编码
encoder = OneHotEncoder(sparse=False)
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)
print(X_test_encoded)

# 训练LR模型
lr_model = LogisticRegression()
lr_model.fit(X_train_encoded, y_train)

# 模型评估
y_pred = lr_model.predict(X_test_encoded)
print("预测结果:", y_pred)


# TOP-N推荐
# user_ids = X_test[:, 0]
# top_n = 2  # 设置TOP-N推荐数量
#
# for user_id, pred_ratings in zip(user_ids, y_pred):
#     # 获取该用户未评分的物品ID
#     user_items = [data[1] for data in train_data if data[0] == user_id]
#     unrated_items = [item for item in range(1, 4) if item not in user_items]
#
#     # 获取TOP-N推荐物品
#     top_n_items = sorted(zip(unrated_items, pred_ratings), key=lambda x: x[1], reverse=True)[:top_n]
#     top_n_items = [item for item, _ in top_n_items]

    # print(f"用户{user_id}的TOP-{top_n}推荐物品:", top_n_items)
