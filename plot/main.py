import data as da
import non_personal as np
import collaborative_filtering as cf
import evaluation as ev
# Import Pandas
import pandas as pd

# Load restaurant data
movies = pd.read_csv('data/movies_subset.csv', low_memory=False)
# Load rating data
rating = pd.read_csv('data/ratings.csv', low_memory=False)
# Load test rating data
test_set = pd.read_csv('data/test.csv', low_memory=False)

cf1 = cf.collaborative_filtering()

# PART 1 - DATA
def analsys(data):
    da.watch_data_info(data)
    da.print_data(data)
    da.plot_data(data)

# PATR 2 - NON PERSONAL RECOMMENDATION SYSTEM
def non_personal_rec(data, k, age):
    np.get_simple_recommendations(data, k, True)
    np.get_simple_age_recommendations(data, age, k, True) #write this function

# PATR 3 - COLLABORATING FILLTERING RECOMMENDATION SYSTEM
def collaborative_filtering_rec(data, similarity, user_based = True):
    global cf1

    if(user_based):
        cf1.create_user_based_matrix(data, similarity)
    else:
        cf1.create_item_based_matrix(data, similarity)


    result = cf1.predict_movies("283225", 5)
    print(result)



# PART 4 - EVALUATION
def evaluate_rec():


    ev.precision_10(test_set,cf1)
    ev.ARHA(test_set,cf1)
    ev.precision_10_avg(test_set, cf1)
    ev.ARHA_avg(test_set, cf1)
    ev.RSME(test_set, cf1)
    print("last part")






def main():


    # rating1 = rating.copy().loc[(rating['rating'] == 4) | (rating['rating'] == 5)]
    # avg = rating.groupby('movieId')['rating'].mean()
    # ratings_per_user = rating.groupby('userId', as_index=False) \
    #     .agg(n_ratings=('movieId', 'count')) \
    #     .query('n_ratings>150 and n_ratings<800') \
    #     .sort_values('n_ratings', ascending=False)
    # print(len(ratings_per_user))
    # ratings3 = rating[rating.userId.isin(ratings_per_user.userId)]
    # ratings3.to_csv('movie_ratings_subset.csv')
    # from sklearn.model_selection import train_test_split
    # train, test = train_test_split(ratings3, test_size=0.3)
    # print(len(test))
    # print(len(train))
    # test.to_csv('test.csv', index=False)
    # train.to_csv('ratings.csv', index=False)
    # avg.to_csv('avg_e.csv')
    #
    print('=====')
    #data
    analsys((rating, movies))

    #non-personalized recommendation system
    # non_personal_rec((rating,movies,users), 5, 31)

    #collaborative filtering
    collaborative_filtering_rec((rating,movies), "cosine")

    #evaluation
    #evaluate_rec()


if __name__ == "__main__":
    main()







