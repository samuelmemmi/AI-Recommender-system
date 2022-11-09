import sys
import matplotlib.pyplot as plt
import seaborn as sns

def watch_data_info(data):
    for d in data:
        # This function returns the first 5 rows for the object based on position.
        # It is useful for quickly testing if your object has the right type of data in it.
        print(d.head())

        # This method prints information about a DataFrame including the index dtype and column dtypes,
        # non-null values and memory usage.
        print(d.info())

        # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a
        # dataset’s distribution, excluding NaN values.
        print(d.describe(include='all').transpose())


def print_data(data):
    ratings = data[0]
    total_rank = ratings.shape[0]

    ratingslist = ratings.values.tolist()
    userIdlist = ratings["userId"].values.tolist()
    movieIdlist = ratings["movieId"].values.tolist()

    # set of unique users, movies ranking
    users_rank_set = {"user"}
    movie_rank_set = {"movie"}

    # dictionary of users/ratings, movies/ratings
    users_dict = {}
    movie_dict = {}

    for r in ratingslist:
        if r[0] in users_dict:
            users_dict[r[0]] += 1
        else:
            users_dict[r[0]] = 1

        if r[1] in movie_dict:
            movie_dict[r[1]] += 1
        else:
            movie_dict[r[1]] = 1

    for u in userIdlist:
        users_rank_set.add(u)

    for m in movieIdlist:
        movie_rank_set.add(m)

    min_rating_movie = movie_dict[min(movie_dict, key=movie_dict.get)]
    max_rating_movie = movie_dict[max(movie_dict, key=movie_dict.get)]
    min_rating_user = users_dict[min(users_dict, key=users_dict.get)]
    max_rating_user = users_dict[max(users_dict, key=users_dict.get)]

    print(len(users_rank_set)-1)
    print(len(movie_rank_set)-1)
    print(total_rank)
    print(min_rating_movie)
    print(max_rating_movie)
    print(min_rating_user)
    print(max_rating_user)

    # sys.exit(1)


def plot_data(data, plot=True):
    ratings = data[0]
    ratingslist = ratings.values.tolist()
    rat_dict = {}
    for r in ratingslist:
        if r[2] in rat_dict:
            rat_dict[r[2]] += 1
        else:
            rat_dict[r[2]] = 1

    rat_list = rat_dict.items()
    rat_list = sorted(rat_list)
    x, y = zip(*rat_list)

    plt.plot(x, y)
    if plot:
        plt.show()
    # sys.exit(1)
