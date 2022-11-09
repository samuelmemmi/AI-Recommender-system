import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
# Import Pandas
import pandas as pd

Arhr = []


def precision_help(test_set, cf, matrix):
    user_list = set(test_set["userId"].values)
    test = test_set.pivot_table("rating", index=["userId"], columns="movieId")
    hit = 0
    rank = 0
    for user in user_list:
        tmp_hit = 0
        ten_movie, rank_d = cf.predict_helper(matrix, user, 10)
        for movie in ten_movie:
            tmp_rank = 0
            rating = test.at[int(user), int(movie)]
            if rating >= 4:
                tmp_hit += 1
                tmp_rank = rank_d[movie]
                tmp_rank = 1 / tmp_rank
            rank += tmp_rank
        tmp_hit = tmp_hit / 10
        hit += tmp_hit
    hit = hit / len(user_list)
    rank = rank / len(user_list)
    Arhr.append(rank)
    return hit


def precision_10(test_set, cf, is_user_based=True):
    if is_user_based:
        hit = precision_help(test_set, cf, cf.user_based_matrix)
    else:
        hit = precision_help(test_set, cf, cf.item_based_matrix)

    print("Precision_k: " + str(hit))


def ARHA(test_set, cf, is_user_based=True):
    val = Arhr[0]
    print("ARHR: " + str(val))


def rsme_help(test_set, cf, matrix):
    test_set_sorted = test_set.sort_values(by=["userId", "movieId"])
    test = test_set_sorted.pivot_table("rating", index=["userId"], columns="movieId")
    actual_rating2D = test.to_numpy()
    user_len = test.shape[0]
    movie_len = test.shape[1]
    diff2 = 0
    n = 0

    for i in range(user_len):
        for j in range(movie_len):
            yi = actual_rating2D[i, j]
            if not pd.isna(yi):
                yhat = matrix[i, j]
                diff1 = yi - yhat
                diff2 += pow(diff1, 2)
                n += 1

    diff2 = diff2 / n
    rsme = sqrt(diff2)
    return rsme


def RSME(test_set, cf, is_user_based=True):
    if is_user_based:
        rsme = rsme_help(test_set, cf, cf.user_based_matrix)
    else:
        rsme = rsme_help(test_set, cf, cf.item_based_matrix)

    print("RSME: " + str(rsme))
