import math
from keras.layers import Embedding
from keras.models import Sequential
import pandas as pd
import numpy as np
import random


def read_100k():
    pd.set_option('display.max_columns', None)  # 显示所有列
    df_user = pd.read_csv('./data/ml-100k/u.user', sep='|',
                          names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    age_bins = [0, 18, 25, 35, 45, 50, 56, 100]
    age_labels = ['0-17', '18-24', '25-34', '25-44', '45-49', '50-55', '56-100']
    df_user['age_bin'] = pd.cut(df_user['age'], bins=age_bins, labels=age_labels)
    age_onehot = pd.get_dummies(df_user['age_bin'])
    gender_onehot = pd.get_dummies(df_user['gender'])
    occupation_onehot = pd.get_dummies(df_user['occupation'])
    user_onehot = pd.concat([df_user[['user_id']], age_onehot, gender_onehot, occupation_onehot], axis=1)

    filename = "./data/ml-100k/u.item"
    rnames = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
              'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    df_item = pd.read_csv(filename, sep='|', header=None, names=rnames, engine='python', encoding='latin1')
    df_item.drop(['title', 'release_date', 'video_release_date', 'IMDb_URL'], axis=1, inplace=True)
    filename = "./data/ml-100k/u.data"
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    df_ratings = pd.read_table(filename, sep='\t', header=None, names=rnames, engine='python')
    df_ratings.drop(['timestamp'], axis=1, inplace=True)
    df_ratings.loc[df_ratings['rating'] < 4, 'rating'] = 0
    df_ratings.loc[df_ratings['rating'] >= 4, 'rating'] = 1
    data = pd.merge(df_ratings, user_onehot, on='user_id')
    data = pd.merge(data, df_item, on='movie_id')
    data = data.sample(frac=1)
    data = data.drop('user_id', axis=1)
    data = data.drop('movie_id', axis=1)
    split_ratio = 1
    cut_off = int(len(data) * split_ratio)
    data_train = data.iloc[:cut_off]
    data_test = data.iloc[cut_off:]
    data.to_csv("FM_model.csv", index=False)
    data_train.to_csv("FM_model_train.csv", index=False)
    data_test.to_csv("FM_model_test.csv", index=False)


def read_1m():
    pd.set_option('display.max_columns', None)
    movies = pd.read_csv('./data/ml-1m/movies.dat', sep='::', names=['movie_id', 'Title', 'Genres'], engine='python',
                         encoding='latin1')
    encoded_genres = pd.get_dummies(movies['Genres'].str.split('|', expand=True).stack(), prefix='Genre').sum(level=0)

    movies_encoded = movies.assign(unknown=0)
    movies_encoded = pd.concat([movies_encoded.drop(['Title', 'Genres'], axis=1), encoded_genres], axis=1)
    new_data = {'movie_id':0, 'unknown':0, 'Genre_Action':0, 'Genre_Adventure':0, 'Genre_Animation':0,
                'Genre_Children\'s':0, 'Genre_Comedy':0, 'Genre_Crime':0, 'Genre_Documentary':0, 'Genre_Drama':0,
                'Genre_Fantasy':0, 'Genre_Film-Noir':0, 'Genre_Horror':0, 'Genre_Musical':0, 'Genre_Mystery':0,
                'Genre_Romance':0, 'Genre_Sci-Fi':0, 'Genre_Thriller':0, 'Genre_War':0, 'Genre_Western':0, }
    new_movie = []
    i = 0
    for movie in movies_encoded['movie_id']:
        i += 1
        while i != movie:
            new_data['movie_id'] = i
            new_row = pd.DataFrame(new_data, index=[i-0.5])
            new_movie.append(new_row)
            i += 1
    for movie in new_movie:
        i = int(movie['movie_id']-1)
        movies_encoded = pd.concat([movies_encoded.iloc[:i], movie, movies_encoded.iloc[i:]]).reset_index(drop=True)
    df_user = pd.read_csv('./data/ml-1m/users.dat', sep='::',names=['user_id', 'gender', 'age', 'occupation',
                                                                    'zip_code'], engine='python',encoding='latin1')
    age_onehot = pd.get_dummies(df_user['age'])
    gender_onehot = pd.get_dummies(df_user['gender'])
    occupation_onehot = pd.get_dummies((df_user['occupation']))
    administrator = occupation_onehot[1]
    artist = occupation_onehot[2]
    doctor = occupation_onehot[6]
    educator = pd.Series(0, index=occupation_onehot.index)
    engineer = occupation_onehot[17]
    entertainment = pd.Series(0, index=occupation_onehot.index)
    executive = occupation_onehot[7]
    healthcare = occupation_onehot[6]
    homemaker = occupation_onehot[9]
    lawyer = occupation_onehot[11]
    librarian = pd.Series(0, index=occupation_onehot.index)
    marketing = occupation_onehot[14]
    none = occupation_onehot[19]
    other = occupation_onehot[0]
    programmer = occupation_onehot[12]
    retired = occupation_onehot[13]
    salesman = occupation_onehot[14]
    scientist = occupation_onehot[15]
    student = occupation_onehot.apply(lambda row: row[4] + row[10], axis=1)
    technician = occupation_onehot[17]
    writer = occupation_onehot[20]
    user_onehot = pd.concat([df_user[['user_id']], age_onehot, gender_onehot, administrator, artist, doctor, educator,
                             engineer, entertainment, executive, healthcare, homemaker, lawyer,librarian, marketing,
                             none, other, programmer, retired, salesman, scientist, student, technician, writer], axis=1)
    filename = "./data/ml-1m/ratings.dat"
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    df_ratings = pd.read_table(filename, sep='::', header=None, names=rnames, engine='python',encoding='latin1')
    df_ratings.drop(['timestamp'], axis=1, inplace=True)
    df_ratings = df_ratings.sort_values(by='rating', ascending=False)  # 先按评分降序排序
    df_ratings = df_ratings.groupby('user_id').head(10)  # 每个用户保留前十条评分数据
    df_ratings.loc[df_ratings['rating'] < 4, 'rating'] = 0
    df_ratings.loc[df_ratings['rating'] >= 4, 'rating'] = 1
    data = pd.merge(df_ratings, user_onehot, on='user_id')
    data = pd.merge(data, movies_encoded, on='movie_id')
    data = data.drop('user_id', axis=1)
    data = data.drop('movie_id', axis=1)
    data.to_csv("FM_1m_model.csv", index=False)
    # user_data = user_onehot.drop('user_id', axis=1).loc[4].tolist()
    # movie_data = movies_encoded.drop('movie_id', axis=1).loc[4].tolist()
    # user_data.extend(movie_data)
    # print(user_data)
    # print(len(user_data))
    return user_onehot, movies_encoded


# 用于fm训练的数据
def read_1m_fm():
    pd.set_option('display.max_columns', None)
    movies = pd.read_csv('./data/ml-1m/movies.dat', sep='::', names=['movie_id', 'Title', 'Genres'], engine='python',
                         encoding='latin1')
    encoded_genres = pd.get_dummies(movies['Genres'].str.split('|', expand=True).stack(), prefix='Genre').sum(level=0)
    movie_onehot = movies.assign(unknown=0)
    movie_onehot = pd.concat([movie_onehot.drop(['Title', 'Genres'], axis=1), encoded_genres], axis=1)
    new_data = {'movie_id': 0, 'unknown': 0, 'Genre_Action': 0, 'Genre_Adventure': 0, 'Genre_Animation': 0,
                'Genre_Children\'s': 0, 'Genre_Comedy': 0, 'Genre_Crime': 0, 'Genre_Documentary': 0, 'Genre_Drama': 0,
                'Genre_Fantasy': 0, 'Genre_Film-Noir': 0, 'Genre_Horror': 0, 'Genre_Musical': 0, 'Genre_Mystery': 0,
                'Genre_Romance': 0, 'Genre_Sci-Fi': 0, 'Genre_Thriller': 0, 'Genre_War': 0, 'Genre_Western': 0, }
    new_movie = []
    i = 0
    for movie in movie_onehot['movie_id']:
        i += 1
        while i != movie:
            new_data['movie_id'] = i
            new_row = pd.DataFrame(new_data, index=[i - 0.5])
            new_movie.append(new_row)
            i += 1
    for movie in new_movie:
        i = int(movie['movie_id'] - 1)
        movie_onehot = pd.concat([movie_onehot.iloc[:i], movie, movie_onehot.iloc[i:]]).reset_index(drop=True)
    df_user = pd.read_csv('./data/ml-1m/users.dat', sep='::', names=['user_id', 'gender', 'age', 'occupation',
                                                                     'zip_code'], engine='python', encoding='latin1')
    age_onehot = pd.get_dummies(df_user['age'])
    gender_onehot = pd.get_dummies(df_user['gender'])
    occupation_onehot = pd.get_dummies((df_user['occupation']))
    user_onehot = pd.concat([df_user[['user_id']], age_onehot, gender_onehot, occupation_onehot],axis=1)
    df_ratings = pd.read_csv('./data/ml-1m/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'],
                             engine='python', encoding='latin1')
    df_ratings.drop(['timestamp'], axis=1, inplace=True)
    # df_ratings.loc[df_ratings['rating'] < 4, 'rating'] = 0
    # df_ratings.loc[df_ratings['rating'] >= 4, 'rating'] = 1

    embedding_dim = 8

    model = Sequential()
    model.add(
        Embedding(input_dim=np.max(user_onehot['user_id']) + 1, output_dim=embedding_dim, input_length=1, name='user_embedding'))

    model.compile(optimizer='adam', loss='mse')

    movie_ids = np.array(movie_onehot['movie_id'])
    user_ids = np.array(user_onehot['user_id'])

    user_embedding = model.predict(user_ids)
    user_embedding_2d = np.squeeze(user_embedding)
    user_onehot = pd.concat([user_onehot, pd.DataFrame(user_embedding_2d)], axis=1)

    model = Sequential()
    model.add(Embedding(input_dim=np.max(movie_onehot['movie_id']) + 1, output_dim=embedding_dim, input_length=1,
                        name='movie_embedding'))
    model.compile(optimizer='adam', loss='mse')
    movie_embedding = model.predict(movie_ids)
    movie_embedding_2d = np.squeeze(movie_embedding)
    movie_onehot = pd.concat([movie_onehot, pd.DataFrame(movie_embedding_2d)], axis=1)

    data = pd.merge(df_ratings, user_onehot, on='user_id')
    data = pd.merge(data, movie_onehot, on='movie_id')

    data = data.drop('user_id', axis=1)
    data = data.drop('movie_id', axis=1)
    data = data.sample(frac=1)
    split_ratio = 0.5
    cut_off = int(len(data) * split_ratio)
    data_train = data.iloc[:cut_off]
    data_test = data.iloc[cut_off:]
    data_train.loc[data_train['rating'] < 4, 'rating'] = 0
    data_train.loc[data_train['rating'] >= 4, 'rating'] = 1
    data.to_csv("FM_model.csv", index=False)
    data_train.to_csv("FM_model_train.csv", index=False)
    data_test.to_csv("FM_model_test.csv", index=False)
    # data_train.to_csv("FM_model_train1.csv", index=False)
    # data_test.to_csv("FM_model_test1.csv", index=False)
    return user_onehot, movie_onehot


def make_test_df():
    movies_test = np.load('A_movies_small_test.npy')
    user_data, movie_data = read_1m()
    data = pd.DataFrame()
    for i, index_array in enumerate(movies_test):
        user_array = []
        for j, movie_rate in enumerate(index_array):
            if movie_rate != 0:
                user = user_data.iloc[i]
                movie = movie_data.iloc[j]
                user_array.append(pd.concat([user, movie], axis=0))
        user_data_concat = pd.concat(user_array, axis=1).T.reset_index(drop=True)
        data = pd.concat([data, user_data_concat], axis=0)
    data.to_csv("1m_result_test.csv", index=False)


if __name__ == '__main__':
    # make_test_df()
    # read_100k()
    read_1m_fm()

