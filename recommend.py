import pandas as pd
import pickle 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from fuzzywuzzy import process

df = pd.read_csv("movies_clean.csv", index_col="movieId")

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

X_tfidf = tfidf.transform(df["tags"])

#recommnd function

def recommend(movie_title, top_k=5):

    # fuzzy match: hitta närmaste titel
    all_titles = df["title"].values
    best_match, score = process.extractOne(movie_title, all_titles)

    if score < 60:
        raise ValueError(f"Hittade ingen film som liknar '{movie_title}'.")

    print(f"Matchade till: {best_match}")

    movie_title = best_match
    movie_id = df[df["title"] == movie_title].index[0]
    movie_tags = df.loc[movie_id, "tags"]

    # vectorize input movie tags
    input_vec = tfidf.transform([movie_tags])

    # cosine similarity
    similarities = cosine_similarity(input_vec, X_tfidf)[0]

    # predict ratings for all movies
    predicted_ratings = model.predict(X_tfidf)

    # combine similarity and predicted rating based on tags
    score = 0.7 * similarities + 0.3 * predicted_ratings 

    # remove input movie
    df["score"] = score
    df_filtered = df.drop(movie_id)

    # take top 50 candidates
    top_50 = df_filtered.nlargest(50, "score")

    # KMeans clustering for diversity
    kmeans = KMeans(n_clusters=top_k, random_state=42)
    row_positions = df.index.get_indexer(top_50.index)
    clusters = kmeans.fit_predict(X_tfidf[row_positions])

    top_50["cluster"] = clusters

    # picking best movie from each cluster
    recommendations = (
        top_50.sort_values("score", ascending=False)
        .groupby("cluster").head(1).sort_values("score", ascending=False)
    )

    return recommendations["title"].tolist()

# run example
if __name__ == "__main__":
    film = input("Skriv en film, med filmens år inom parantes: ")
    try:
        recs = recommend(film)
        print("\nRekommendationer:\n")
        for film in recs:
            print(f"- {film}")
    except ValueError as e:
        print(e)