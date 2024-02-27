import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from deepctr.models import WDL
from deepctr.feature_column import SparseFeat, get_feature_names

# 读取数据
filename = "./data/ml-1m/ratings.dat"
rname = ['user_id', 'movie_id', 'rating', 'timestamp']
df = pd.read_table(filename, sep='::', header=None, names=rname, engine='python')

# 删除'timestamp'列
df.drop(['timestamp'], axis=1, inplace=True)

# 洗牌数据
data = df.sample(frac=1)

# 数据预处理
sparse_features = ["movie_id", "user_id"]  # 不包括'timestamp'列
target = ['rating']

# 特征数值化
for feat in sparse_features:
    transfor = LabelEncoder()
    data[feat] = transfor.fit_transform(data[feat])

# 生成词向量
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique()) for feat in sparse_features]

# 定义线性和深度部分的特征列
linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 将数据集切分成训练集和测试集
train, test = train_test_split(data, test_size=0.2)

# 将数据转换为模型输入格式
train_model_input = {name: train[name].values for name in feature_names}
test_model_input = {name: test[name].values for name in feature_names}

# 模型训练
model = WDL(linear_feature_columns, dnn_feature_columns, task='regression')
model.compile("adam", "mse", metrics=['mse'])
history = model.fit(train_model_input, train[target].values, batch_size=256, epochs=10, verbose=True, validation_split=0.2)

# 使用模型进行预测
pred = model.predict(test_model_input, batch_size=256)

# 定义函数来获取每个用户的topN推荐列表
def get_top_n_recommendations(pred, test):
    top_n_recommendations = {}
    k = 10
    for user_id in test['user_id'].unique():
        user_data = test[test['user_id'] == user_id][['user_id', 'movie_id']]
        user_data = user_data.drop_duplicates(subset=['user_id', 'movie_id'])  # 去重
        user_idx = user_data.index  # 获取用户数据在测试集中的索引
        if len(user_idx) == 0:
            continue  # 如果用户在测试集中没有数据，则跳过
        if max(user_idx) >= len(pred):
            continue  # 如果用户数据的最大索引超出了预测数组的范围，则跳过
        user_data['rating_pred'] = pred[user_idx]  # 使用模型预测的评分值
        top_n_recommendations[user_id] = user_data.sort_values(by='rating_pred', ascending=False).head(k)['movie_id'].tolist()
    return top_n_recommendations


# 获取topN推荐列表
top_n_recommendations = get_top_n_recommendations(pred, test)

# 计算准确率和召回率
def calculate_precision_recall_rr(test_data, top_n_recommendations):
    precision_total = 0
    recall_total = 0
    reciprocal_rank_total = 0
    num_users = len(top_n_recommendations)

    for user_id, recommendations in top_n_recommendations.items():
        relevant_items = test_data[(test_data['user_id'] == user_id) & (test_data['rating'] >= 4)]['movie_id'].tolist()
        intersection = len(set(recommendations) & set(relevant_items))
        precision = intersection / len(recommendations) if len(recommendations) != 0 else 0
        recall = intersection / len(relevant_items) if len(relevant_items) != 0 else 0

        reciprocal_rank = 0
        if intersection > 0:
            first_relevant_index = min(
                (recommendations.index(movie_id) for movie_id in relevant_items if movie_id in recommendations),
                default=-1)
            if first_relevant_index != -1:
                reciprocal_rank = 1 / (first_relevant_index + 1)

        precision_total += precision
        recall_total += recall
        reciprocal_rank_total += reciprocal_rank

    avg_precision = precision_total / num_users if num_users != 0 else 0
    avg_recall = recall_total / num_users if num_users != 0 else 0
    avg_reciprocal_rank = reciprocal_rank_total / num_users if num_users != 0 else 0

    return avg_precision, avg_recall, avg_reciprocal_rank


precision, recall, reciprocal_rank = calculate_precision_recall_rr(test, top_n_recommendations)
print("Average Precision:", precision)
print("Average Recall:", recall)
print("Average Reciprocal Rank:", reciprocal_rank)




