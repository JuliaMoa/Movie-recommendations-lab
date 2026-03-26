import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostRegressor 
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import pickle # python's way to save object to file
from nltk.corpus import stopwords
import nltk 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import spearmanr

# load cleaned data set
df = pd.read_csv("movies_clean.csv", index_col="movieId")

# extract features and target. 
X_text = df["tags"]
y = df["rating"]

# create and fit TF-IDF vectorizer on #tags#   #bigrams = 2-word sequences
tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=2000, lowercase=True) 

X_tfidf = tfidf.fit_transform(X_text)

# train-test-split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Use a smaller sample for cross-validation
X_cv = X_train[:3000]
y_cv = y_train[:3000]

# model = ExtraTreesRegressor() -- was too slow when training
# model = AdaBoostRegressor() -- got slightly worse rmse

model = AdaBoostRegressor()

# hyper parameter search space
param_distribs = {
    "n_estimators": [50, 100, 150, 200, 300],
    "learning_rate": [0.01, 0.03, 0.05, 0.1]
}

rnd_search = RandomizedSearchCV(
                                estimator=model, 
                                param_distributions=param_distribs, 
                                n_iter=10, 
                                scoring="neg_root_mean_squared_error", 
                                cv=3, 
                                n_jobs=-1)

rnd_search.fit(X_cv, y_cv)

print("Best parameters:", rnd_search.best_params_)
print("Best CV RMSE:", -rnd_search.best_score_)

# get the best model
best_model = rnd_search.best_estimator_
# train the best model on the whole training set
best_model.fit(X_train, y_train) 

# predict on the test data
y_pred = best_model.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {test_rmse:.4f}")

#feature importance
feature_names = tfidf.get_feature_names_out()
importances = best_model.feature_importances_
indices = importances.argsort()[::-1]

print("\nTop 20 most important features:")
for i in range(20):
    print(feature_names[indices[i]], importances[indices[i]])

# save TF-IDF model
with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# save trained regression model
with open("model.pkl", "wb") as f:  
    pickle.dump(best_model, f) 

print("Training completed. Models saved as tfidf.pkl and model.pkl")