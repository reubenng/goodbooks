from surprise import Reader, Dataset
import pandas as pd

# 
dataset_dir = '/Users/daniellee/Desktop/University/Data Mining/goodbooks/data/ratings.csv'

reader = Reader(line_format='user item rating', sep=',')

data = Dataset.load_from_file(dataset_dir, reader=reader)

data.split(n_folds=5)

from surprise import SVD, evaluate
algo = SVD()
evaluate(algo, data, measures=['RMSE', 'MAE'])

sim_options = {'name': 'cosine', 'user-based': True}