import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostRegressor 
from sklearn.model_selection import train_test_split
import pickle # python's way to save object to file
from nltk.corpus import stopwords
import nltk 

nltk.download("stopwords")

# load cleaned data set
df = pd.read_csv("movies_clean.csv", index_col="movieId")

# extract features and target. 
X_text = df["tags"]
y = df["rating"]

# create and fit TF-IDF vectorizer on #tags# 
tfidf = TfidfVectorizer(stop_words="english")

X_tfidf = tfidf.fit_transform(X_text)

# train-test-split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# train AdaBoostRegressor on the training data (vectorized features and ratings)
# under the surface it is Decision Forest Stumps
model = AdaBoostRegressor(n_estimators=200) # 200 weak models, stumps of one level trees

model.fit(X_train, y_train)

# save TF-IDF model
with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# save trained regression model
with open("model.pkl", "wb") as f:  
    pickle.dump(model, f) 

print("Training completed. Models saved as tfidf.pkl and model.pkl")