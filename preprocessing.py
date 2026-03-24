import pandas as pd

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv",
                      engine="pyarrow",
                      dtype={"userId": "int32", "movieId": "int32"})

tags = pd.read_csv("tags.csv")

# putting all the tags for one movie into one text string per movie
tags_grouped = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x.dropna()))

# removing the vertical lines, and combining tags and genres
movies["genres_text"] = movies["genres"].str.replace("|", " ")
movies = movies.merge(tags_grouped, on="movieId", how="left")
movies["tags"] = (movies["genres_text"].fillna("")+" "+ movies["tag"].fillna(""))

# group by movie - then pick rating - then take the mean of the ratings per movie
ratings_grouped = ratings.groupby("movieId")["rating"].mean()

# merging in the average rating for each movie
movies = movies.merge(ratings_grouped, on="movieId", how="left")

movies["tags"] = movies["tags"].str.lower()

# picking which columns we keep
df = movies[["movieId", "title", "tags", "rating"]]
df = df.set_index("movieId")

# remove movie-rows with missing ratings
df = df.dropna(subset=["rating"])

# save as new data file
df.to_csv("movies_clean.csv", index=True) 


# split and put away the testing data


# preprocessing pipeline



# in the model pipeline put preprocessing, the tf-idf-vectorizer

# cross-validation and finetuning of hyperparametres

# testing 

# increase diversity with clustering 



