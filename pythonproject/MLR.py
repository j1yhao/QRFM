import numpy as np
from deepctr.models import MLR
from deepctr.feature_column import SparseFeat
from sklearn.metrics import accuracy_score, recall_score

# 加载训练集和测试集评分矩阵
train_data = np.load('A_movies_small.npy')
test_data = np.load('A_movies_small_test.npy')

# 训练集评分矩阵的行数表示用户数量，列数表示物品数量
num_users, num_items = train_data.shape

# 将评分矩阵中的评分>=4的视为用户喜欢的电影，转换为二进制标签
train_labels = (train_data >= 4).astype(int)
test_labels = (test_data >= 4).astype(int)

# 定义用户特征列和物品特征列
user_feature_columns = [SparseFeat(f'user_{i}', 1) for i in range(num_users)]
item_feature_columns = [SparseFeat(f'item_{i}', 1) for i in range(num_items)]

# 创建 MLR 模型
model = MLR(user_feature_columns + item_feature_columns)
model.compile("adam", "binary_crossentropy")

# 拆分训练集数据，用户特征和物品特征
user_features = train_data[:, :num_users]  # 用户特征
item_features = train_data[:, num_users:]  # 物品特征

# 训练模型
model.fit([user_features, item_features], train_labels, batch_size=256, epochs=10, verbose=True)

# 使用模型预测测试集评分
test_user_features = test_data[:, :num_users]  # 测试集用户特征
test_item_features = test_data[:, num_users:]  # 测试集物品特征
predictions = model.predict([test_user_features, test_item_features])

# 根据预测结果生成推荐列表
N = 10  # topN推荐
top_n_recommendations = []
for pred in predictions:
    top_n = np.argsort(pred)[::-1][:N]  # 取概率值最高的前 N 个物品作为推荐结果
    top_n_recommendations.append(top_n)

# 计算准确率和召回率
accuracy = accuracy_score(test_labels.flatten(), (predictions >= 0.5).astype(int).flatten())
recall = recall_score(test_labels.flatten(), (predictions >= 0.5).astype(int).flatten())

print("准确率:", accuracy)
print("召回率:", recall)
