# 🎬 Movie Recommendation System

## 📌 Project Overview

The Movie Recommendation System is a machine learning project that recommends movies to users based on the similarity between movies and user rating patterns.

The system uses a hybrid recommendation approach that combines:

* Content-Based Filtering
* Collaborative Filtering

The project is developed using Python and Streamlit and uses the MovieLens 1M dataset.


## 🎯 Objective

The main objective of this project is to develop a movie recommendation system that can suggest movies similar to a movie selected by the user.

The system analyzes movie genres and user ratings to generate relevant recommendations.



## 📊 Dataset

The project uses the MovieLens 1M Dataset.

The dataset contains:

* 3,883 movies
* 1,000,209 ratings
* 6,040 users

The dataset files used are:

* `movies.dat`
* `ratings.dat`
* `users.dat`



## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit
* TF-IDF
* Cosine Similarity

## ⚙️ Methodology

### 1. Data Loading

The MovieLens dataset is loaded using Pandas.

The three files used are:

* Movies
* Ratings
* Users

### 2. Data Preprocessing

Movie genres are cleaned and converted into a suitable text format for analysis.

The rating data is converted into a movie-user rating matrix.

### 3. Content-Based Filtering

TF-IDF is used to represent movie genres numerically.

Cosine similarity is then used to measure the similarity between movies based on their genres.

### 4. Collaborative Filtering

A movie-user rating matrix is created from the ratings dataset.

Cosine similarity is used to identify movies that have similar user rating patterns.

### 5. Hybrid Recommendation

The final recommendation system combines both approaches.

The final score is calculated using:

Final Score = 0.4 × Content Similarity + 0.6 × Collaborative Similarity

The system then selects the highest-scoring movies and displays the requested number of recommendations.


## 🎬 Example

For the movie:

Toy Story (1995)

The system can recommend:

1. Toy Story 2 (1999)
2. Bug's Life, A (1998)
3. Aladdin (1992)
4. Chicken Run (2000)
5. Antz (1998)



## 💻 Project Structure

Movie_system/

├── movie_recommendation.py
├── app.py
├── movies.dat
├── ratings.dat
├── users.dat
├── requirements.txt
└── README.md



## ▶️ How to Run the Project

### Step 1: Install the required libraries

pip install -r requirements.txt


### Step 2: Run the Streamlit application

python -m streamlit run app.py


### Step 3: Use the application

Select a movie from the dropdown menu, choose the number of recommendations, and click the Recommend Movies button.



## 📈 Results

The developed system successfully generates movie recommendations by combining movie genre information with user rating patterns.

The Streamlit interface allows users to interact with the recommendation system and receive recommendations for their selected movie.


## 🔮 Future Enhancements

* Add movie posters and images
* Add user-based personalized recommendations
* Add movie rating prediction
* Improve recommendation accuracy
* Add user login and rating history
* Deploy the application online



## 📚 Conclusion

The Movie Recommendation System demonstrates how machine learning techniques can be used to build a practical recommendation engine.

By combining content-based filtering and collaborative filtering, the system provides movie recommendations through an easy-to-use Streamlit interface.
