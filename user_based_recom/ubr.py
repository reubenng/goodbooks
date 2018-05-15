import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def normalize_dataset(df):
    # replace 0s to None to calculate mean
    df = df.replace(0, df.replace([0], [None]))

    # subtract mean to all values
    df = df.sub(df.mean(axis=1),axis=0)

    # replace Nones to 0s back again
    df = df.fillna(0)
    return df

def cosine_sim_matrix(df):
    return cosine_similarity(df)

def ranking_matrix(cosine_sim_matrix):
    ranking_matrix = []
    
    np.fill_diagonal(cosine_sim_matrix, -5)

    cosine_sim_matrix = cosine_sim_matrix.tolist()

    for row_i, row in enumerate(cosine_sim_matrix):
        per_row = []
        for el_i, el in enumerate(row):
            if el_i == len(cosine_sim_matrix) - 1:
                continue

            highest_sim_index = cosine_sim_matrix[row_i].index(max(cosine_sim_matrix[row_i]))
            per_row.append(highest_sim_index)
            cosine_sim_matrix[row_i][highest_sim_index] = -5

        ranking_matrix.append(per_row)
    
    ranking_matrix = np.array(ranking_matrix)

    return ranking_matrix

def books_to_predict_matrix(data_df):
    data_matrix = data_df.as_matrix()
    data_matrix = data_matrix.tolist()

    books_to_predict = []

    for row_i, row in enumerate(data_matrix):
        by_user = []
        for el_i, el in enumerate(row):
            if data_matrix[row_i][el_i] == 0:
                by_user.append(el_i)
        
        books_to_predict.append(by_user)

    return books_to_predict

def predict_ratings_top_n(n, data_df, ranking_matrix, books_to_predict):
    data_df = data_df.as_matrix()
    for user_i, user in enumerate(books_to_predict):
        for book_i, book in enumerate(user):
            # predict using top n users
            total_ratings_of_n_users = 0
            for i in range(n):
                total_ratings_of_n_users = total_ratings_of_n_users + data_df[ranking_matrix[user_i][i]][book]

            total_ratings_of_n_users = total_ratings_of_n_users / n
            data_df[user_i][book] = total_ratings_of_n_users

    return data_df

n_users = 10

# data is a dict
data = {'Book1': [random.randint(0,5) for _ in range(n_users)], 'Book2': [random.randint(0,5) for _ in range(n_users)], 'Book3': [random.randint(0,5) for _ in range(n_users)], 'Book4': [random.randint(0,5) for _ in range(n_users)], 'Book5': [random.randint(0,5) for _ in range(n_users)], 'Book6': [random.randint(0,5) for _ in range(n_users)], 'Book7': [random.randint(0,5) for _ in range(n_users)]}

data_df = pd.DataFrame(data=data)
original_df = data_df.copy()
print('[Original Data]\n')
print(data_df)
print('\n')

norm_data_df = normalize_dataset(data_df)
print('[Normalized Data]\n')
print(norm_data_df)
print('\n')

cosine_sim = cosine_similarity(norm_data_df)
print('[Cosine Similarities]\n')
print(cosine_sim)
print('\n')

ranking_matrix = ranking_matrix(cosine_sim)
print('[Similarities Ranking (Ascending order)]\n')
print(ranking_matrix)
print('\n')

books_to_predict = books_to_predict_matrix(data_df)
print('[Books that need to be rated]\n')
print(books_to_predict)
print('\n')

rating_predictions = predict_ratings_top_n(3, data_df, ranking_matrix, books_to_predict)
print('[Ratings Predictions]\n')
print(rating_predictions)
# print(pd.DataFrame(rating_predictions))
print('\n')

print('[Original Ratings]\n')
print(original_df.as_matrix())

# group by method to group data by users
