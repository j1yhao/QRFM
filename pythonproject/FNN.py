import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


train_data = pd.read_csv("./FM_model_train1.csv")
x_train = train_data.drop('rating', axis=1)
x_train = x_train.drop('user_id', axis=1)
x_train = x_train.drop('movie_id', axis=1)
y_train = train_data['rating']


# 定义模型结构
def create_fnn_model(feature_dim):
    model = tf.keras.Sequential([
        Dense(64, activation='relu', input_dim=feature_dim),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model


feature_dim = 65  # 根据实际情况设置特征维度
model = create_fnn_model(feature_dim)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=32)
# # 保存模型
model.save('fnn_model.h5')

model = tf.keras.models.load_model('fnn_model.h5')


test_data = pd.read_csv("./FM_model_test1.csv")
x_test = test_data
y_test = test_data['rating']


user_ids = range(6000)
k = 10
acc_count = 0
count = 0
for user_id in user_ids:
    user_data = x_test[x_test['user_id'] == user_id]
    if user_data.empty:
        continue
    test = user_data.drop(['user_id', 'movie_id', 'rating'], axis=1)
    test = np.array(test)
    predictions = model.predict(test)
    predictions = predictions.flatten()
    top_movies_indices = predictions.argsort()[::-1][0:k]
    top_movies_ids = user_data.iloc[top_movies_indices]["movie_id"].tolist()
    accuracy = 0
    for j in top_movies_ids:
        rating = user_data[user_data['movie_id'] == j]['rating'].values[0]
        if rating >= 4:
            accuracy += 1
    print(accuracy/k)
    acc_count += accuracy/k
    count += 1
print(acc_count/count)

