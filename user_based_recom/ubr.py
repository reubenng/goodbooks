import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def normalize_dataset(df):
    df = df.replace(0, df.replace([0], [None]))
    # print(matrix)
    df = df.sub(df.mean(axis=1),axis=0)
    # print(matrix)
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

n_users = 7

# data is a dict
data = {'Book1': [random.randint(0,5) for _ in range(n_users)], 'Book2': [random.randint(0,5) for _ in range(n_users)], 'Book3': [random.randint(0,5) for _ in range(n_users)], 'Book4': [random.randint(0,5) for _ in range(n_users)], 'Book5': [random.randint(0,5) for _ in range(n_users)], 'Book6': [random.randint(0,5) for _ in range(n_users)], 'Book7': [random.randint(0,5) for _ in range(n_users)]}

data_df = pd.DataFrame(data=data)

norm_data_df = normalize_dataset(data_df)
print(norm_data_df)
print('\n')

cosine_sim = cosine_similarity(norm_data_df)
print(cosine_sim)
print('\n')

ranking_matrix = ranking_matrix(cosine_sim)
print(ranking_matrix)

