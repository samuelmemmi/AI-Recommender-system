import sys
import pandas as pd
import numpy as np
import heapq
from sklearn.metrics.pairwise import pairwise_distances


class collaborative_filtering:
    def __init__(self):
        self.user_based_matrix = []
        self.item_based_matrix = []
        self.ratings = pd.DataFrame()
        self.rating_table = pd.DataFrame()
        self.user_idx = []
        self.movie_idx = []
        self.movie_csv = pd.DataFrame()

    def create_fake_user(self, rating):
        fake = [[283238, 1, 3.5], [283238, 2, 4.5], [283238, 19, 4.5], [283238, 44, 4.0],
                [283238, 107, 3.0], [283238, 158, 4.0], [283238, 165, 2.0],
                [283238, 208, 2.5], [283238, 231, 4.5], [283238, 261, 1.5],
                [283238, 329, 4.0], [283238, 428, 1.5], [283238, 431, 0.5]]
        fake_frame = pd.DataFrame(fake, columns=["userId", "movieId", "rating"])
        rating = pd.concat([rating, fake_frame], ignore_index=True, axis=0)
        return rating

    def ratings_diff(self, data):
        self.movie_csv = data[1]
        rating = self.create_fake_user(data[0])

        self.ratings = rating.sort_values(by=["userId", "movieId"])
        self.rating_table = self.ratings.pivot_table("rating", index=["userId"], columns="movieId")

        self.user_idx = np.array(self.rating_table.index)
        self.movie_idx = np.array(self.rating_table.columns)

        ratings_np = self.rating_table.to_numpy()
        mean_user_rating = self.rating_table.mean(axis=1).to_numpy().reshape(-1, 1)
        ratings_diff = (ratings_np - mean_user_rating)
        ratings_diff[np.isnan(ratings_diff)] = 0

        return ratings_diff, mean_user_rating

    def create_user_based_matrix(self, data):
        ratings_diff, mean_user_rating = self.ratings_diff(data)
        user_similarity = 1 - pairwise_distances(ratings_diff, metric='cosine')
        pd.DataFrame(user_similarity)
        pd.DataFrame(user_similarity.dot(ratings_diff))

        self.user_based_matrix = mean_user_rating + user_similarity.dot(ratings_diff) / np.array(
            [np.abs(user_similarity).sum(axis=1)]).T

    def create_item_based_matrix(self, data):
        ratings_diff, mean_user_rating = self.ratings_diff(data)
        item_similarity = 1 - pairwise_distances(ratings_diff.T, metric='cosine')
        pd.DataFrame(item_similarity)

        self.item_based_matrix = mean_user_rating + ratings_diff.dot(item_similarity) / np.array(
            [np.abs(item_similarity).sum(axis=1)])

    def predict_helper(self, matrix, user_id, k):
        movie_dict = {}
        rank_dict = {}
        i = 1
        user_id_index = np.where(self.user_idx == int(user_id))
        user_id_index = user_id_index[0].item(0)
        ratings_of_user = matrix[user_id_index]

        while len(movie_dict) < k:
            maxx = ratings_of_user.max()
            movie_id_idx = np.where(ratings_of_user == maxx)
            movie_id_idx = movie_id_idx[0].item(0)
            movie_id = self.movie_idx[movie_id_idx]
            ratings_of_user[movie_id_idx] = 0
            already_seen = self.rating_table.at[int(user_id), int(movie_id)]
            if pd.isna(already_seen):
                movie = self.movie_csv.loc[self.movie_csv["movieId"] == movie_id]
                movie = movie.values
                movie_dict[str(movie_id)] = movie
                rank_dict[str(movie_id)] = i
                i += 1
        return movie_dict, rank_dict

    def predict_movies(self, user_id, k, is_user_based=True):
        if is_user_based:
            predict_list, rank = self.predict_helper(self.user_based_matrix, user_id, k)
        else:
            predict_list, rank = self.predict_helper(self.item_based_matrix, user_id, k)
        return predict_list