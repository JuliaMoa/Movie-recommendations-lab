explaining which methods you use, 
what limitations apply,
and what choices you have made. 
-- chose to use mean-rating
-- choose to include genres in tags
-- creating the data set this way -- to able to check correlation tags and ratings

## INTRODUCTION

The purpose of this project was to create a movie recommendation system that takes one movie as input and gives 5 recommended movies. A content based recommender approach was chosen using several different methods. The regression model used will be explained in another section.

#### TF-IDF
TF-IDF is a weighting scheme used to represent text numerically. It increases the weight of words that are important in a specific document while down-weighting words that are common across the entire corpus. 

![alt text](image-3.png)
![alt text](image-4.png)
where N is the number of documents and df(t) is the number of documents containing the term t. 

TF-IDF weight: TF-IDF(t,d) = TF(t,d) * IDF(t)

TF-IDF transforms text into a high-dimensional vector space where each dimension corresponds to a word or n-gram.

#### COSINE SIMILARITY

Cosine similarity measures how similar two documents are by comparing the angle between their TF-IDF vectors. It works well for sparse TF-IDF vectors because it focuses on relative word usage. 

![alt text](image.png) 

Value ranges from 0 to 1. 1 means identical direction (high similarity). 0 means othogonal (no similarity)

### K-Means Clustering
This is an unsuperivsed learning algorithm that partitions data into k clusters by minimizing the distance between points and their assigned cluster centers. When applied to TF-IDF vectors, k-means groups documents based on similarity in word usage. 

![alt text](image-1.png)

where:
![alt text](image-2.png)

Algorithm steps:
1. Initialize 𝑘 centroids
2. Assign each point to the nearest centroid
3. Update centroids as the mean of assigned points
4. Repeat until convergence

## EXPLORATORY DATA ANALYSIS

To explore wheter tags relate to movie ratings, the dataset was split into high-rated (rating > 4.0) and low-rated (rating < 3.0) films. A simple frequency analysis showed that high rated movies contained more quality- and mood-related words such as great, good, classic, Oscar, atmopheric and thought-provoking, while low-rated movies used more generic or negatively associated terms. Genres were common for both,comedy common in both, fantasy and sci-fi in the high-rated films and horror, crime and romance were common tags in the low rated movies.

This difference suggested that tags carry meaningful information about perceived movie quality. The regression model later confirmed this: the most important features were the same value-laden and genre-related words identified in the frequency analysis. 

## MODEL

Tested and evaluated GradientBoostingRegression with hyperparameters...
As well as ExtraTrees with hyperparameters...
Both were too slow.

AdaBoostRegressor was chosen as a value regressor to predict ratings.

--- beskriv/modell algoritmen och de

--- algoritmer/modell du använd samt vilka hyper-parametrar du använt. nämn alla modeller du använder, men beskriv i detalj bara den du faktiskt använder i slutändan (den som var bäst)

How the model is used with the other methods?

## RESULTS
--- utvärdera modellen. du kan jämföra modeller tex i en tabell, men i löptexten beskriv bara den du använde i slutändan. Beskriv resultaten både i ord och i en figur eller python output. 

The AdaBoostRegressor was chosen for the project.  
Hyperparameters:
ested with RMSE and depending on the run, scored a

The model is value-regressor of meanrating (0-5). 
RMSE measures how well the model predicts the rating. Using cross validation rmse on the training data, the model scores around 0.7. This means in average the model missed with 0.78 rating units. The cv-standard deviation was around 0.02, meaning that the model is robust. 

Performing rmse on the test-data the result was very close. Meaning the model performs as expected compared to the CV-RMSE. Indicates no overfitting.

Predicting rating is of course only one part of the full method.

The resulting recommendations of the system were quite good. 

## DISCUSSION
--- förklara kortakommanden och/eller begränsningar och resonera kring vad resultatet innebär för problemställningen

The somewhat high cv-rmse of the AdaBoostRegressor-model can be understand in the way that the movie tags cannot completely explain the variation in the movies' average grade. But together with the other methods and techniques in the full recommendation system, it does work quite well. 

