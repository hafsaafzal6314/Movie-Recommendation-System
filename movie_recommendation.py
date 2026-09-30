import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Movies
movies = pd.read_csv(
    "movies.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["movieId", "title", "genres"]
)

# Load Ratings
ratings = pd.read_csv(
    "ratings.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["userId", "movieId", "rating", "timestamp"]
)

# Load Users
users = pd.read_csv(
    "users.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["userId", "gender", "age", "occupation", "zipCode"]
)

print("Movies:", movies.shape)
print("Ratings:", ratings.shape)
print("Users:", users.shape)
#step1: checking the data
print("\nMovies Dataset:")
print(movies.head())

print("\nRatings Dataset:")
print(ratings.head())

print("\nUsers Dataset:")
print(users.head())
#step2: Data cleaning & preprocessing
# Check for missing values
print("\nMissing values in Movies:")
print(movies.isnull().sum())

print("\nMissing values in Ratings:")
print(ratings.isnull().sum())

print("\nMissing values in Users:")
print(users.isnull().sum())
# Prepare genres for content-based filtering
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

print("\nUpdated Genres:")
print(movies[["title", "genres"]].head())
# Convert genres into numerical features
tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(movies["genres"])

print("\nTF-IDF Matrix Shape:")
print(tfidf_matrix.shape)
# Calculate similarity between movies
similarity_matrix = cosine_similarity(tfidf_matrix)

print("\nSimilarity Matrix Shape:")
print(similarity_matrix.shape)
# Recommendation function
def recommend_movies(movie_title, num_recommendations=5):
    movie_index = movies[movies["title"] == movie_title].index[0]

    similarity_scores = list(enumerate(similarity_matrix[movie_index]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in similarity_scores[1:num_recommendations + 1]:
        recommendations.append(movies.iloc[index]["title"])

    return recommendations


# Test the recommendation system
movie = "Toy Story (1995)"

recommendations = recommend_movies(movie)

print("\nRecommendations for:", movie)

for i, recommendation in enumerate(recommendations, 1):
    print(f"{i}. {recommendation}")
    
    
# Create a user-movie rating matrix
rating_matrix = ratings.pivot_table(
    index="movieId",
    columns="userId",
    values="rating"
)

print("\nRating Matrix Shape:")
print(rating_matrix.shape)

# Fill missing ratings with 0
rating_matrix_filled = rating_matrix.fillna(0)

# Calculate similarity between movies based on user ratings
rating_similarity = cosine_similarity(rating_matrix_filled)

print("\nRating Similarity Matrix Shape:")
print(rating_similarity.shape)

# Collaborative filtering recommendation function
def collaborative_recommend(movie_title, num_recommendations=5):
    # Get movie ID
    movie_id = movies[movies["title"] == movie_title]["movieId"].values[0]

    # Find the movie's position in the rating matrix
    movie_index = rating_matrix.index.get_loc(movie_id)

    # Get similarity scores
    similarity_scores = list(
        enumerate(rating_similarity[movie_index])
    )

    # Sort by similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, score in similarity_scores[1:]:
        recommended_movie_id = rating_matrix.index[index]

        movie_title_result = movies[
            movies["movieId"] == recommended_movie_id
        ]["title"].values

        if len(movie_title_result) > 0:
            recommendations.append(movie_title_result[0])

        if len(recommendations) == num_recommendations:
            break

    return recommendations

# Test collaborative filtering
collaborative_results = collaborative_recommend("Toy Story (1995)")

print("\nCollaborative Recommendations for: Toy Story (1995)")

for i, movie in enumerate(collaborative_results, 1):
    print(f"{i}. {movie}")

# Combined recommendation function
def combined_recommend(movie_title, num_recommendations=5):
    movie_index = movies[movies["title"] == movie_title].index[0]

    # Content-based similarity scores
    content_scores = similarity_matrix[movie_index]

    # Movie ID
    movie_id = movies.iloc[movie_index]["movieId"]

    # Check if movie exists in rating matrix
    if movie_id in rating_matrix.index:
        rating_index = rating_matrix.index.get_loc(movie_id)
        collaborative_scores = rating_similarity[rating_index]

        # Create dictionary for collaborative scores
        collaborative_dict = dict(
            zip(rating_matrix.index, collaborative_scores)
        )
    else:
        collaborative_dict = {}

    # Calculate final scores
    final_scores = []

    for i, row in movies.iterrows():
        current_movie_id = row["movieId"]

        content_score = content_scores[i]

        collaborative_score = collaborative_dict.get(
            current_movie_id, 0
        )

        final_score = (
            0.4 * content_score +
            0.6 * collaborative_score
        )

        final_scores.append((i, final_score))

    # Sort by final score
    final_scores = sorted(
        final_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Get recommendations
    recommendations = []

    for index, score in final_scores:
        if index != movie_index:
            recommendations.append(movies.iloc[index]["title"])

        if len(recommendations) == num_recommendations:
            break

    return recommendations

# Test combined recommendation system
if __name__ == "__main__":
    final_results = combined_recommend("Toy Story (1995)")

    print("\nFinal Top 5 Recommendations for: Toy Story (1995)")
    for i, movie in enumerate(final_results, 1):
        print(f"{i}. {movie}")