### Brief description
A content‑based movie recommender that suggests 5 movies based on a single input title. The system uses TF‑IDF, cosine similarity, AdaBoost regression, and k‑means clustering to produce relevant and diverse recommendations.

### Data requirements
This project uses the MovieLens dataset and can be downloaded from the link https://grouplens.org/datasets/movielens/ 

Under “recommended for education and development” download ml-latest.zip

Required files (not included in this repo):

- movies.csv

- ratings.csv

- tags.csv

Place all CSV files in the project root before running the code.

These are combined during preprocessing into:

movies_clean.csv — containing movieId, title, tags, and mean rating.

### How to use the recommendation system:

Run recommend.py, then enter a movie title in the terminal.


