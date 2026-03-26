explaining which methods you use, 
what limitations apply,
and what choices you have made. 

INTRODUCTION

The purpose of this project was to create a movie recommendation system that takes one movie as input and gives 5 recommended movies. 

--- relevanta formler och definitioner


- td-idf är en matematisk viktning som mäter hur viktigt ett ord är jämfört med hela corpuset. skapar textvektorer. 
- cosine similarity is a mathematical operation used , används för att mäta likhet mellan tf-idf vektorer ---finns formel och definition - mäter vinkeln mellan vektorer! 
- AdaBoostRegressor - ensemblemodell som gör en viktag median av svaga modeller för att prediktera filmkvalitet. 
- KMean - unsupervised ML-modell som hittar mönster i data, delar upp i kluster och representerar varje kluster med ett "center" (finns matematisk formel)

EXPLORATORY DATA ANALYSIS

    --samband i datamängden --- bara det jag använder

MODEL

Tested and evaluated GradientBoostingRegression with hyperparameters...

AdaBoost
--- beskriv/modell algoritmen och de


Kmean

--- algoritmer/modell du använd samt vilka hyper-parametrar du använt. nämn alla modeller du använder, men beskriv i detalj bara den du faktiskt använder i slutändan (den som var bäst)

This is a content-based recommender system. 
Predicting ratings from tags


RESULTS
--- utvärdera modellen. du kan jämföra modeller tex i en tabell, men i löptexten beskriv bara den du använde i slutändan. Beskriv resultaten både i ord och i en figur eller python output. 


The AdaBoostRegressor was chosen for the project.  
Hyperparameters:
ested with RMSE and depending on the run, scored a

The model is value-regressor of meanrating (0-5). 
RMSE measures how well the model predicts the rating. Using cross validation rmse on the training data, the model scores around 0.7. This means in average the model missed with 0.78 rating units. The cv-standard deviation was around 0.02, meaning that the model is robust. 

Performing rmse on the test-data the result was very close. Meaning the model performs as expected compared to the CV-RMSE. Indicates no overfitting.

Predicting rating is of course only one part of the full method.

Results of the full pipeline:

DISCUSSION
--- förklara kortakommanden och/eller begränsningar och resonera kring vad resultatet innebär för problemställningen

The somewhat high cv-rmse of the AdaBoostRegressor-model can be understand in the way that the movie tags cannot completely explain the variation in the movies' average grade. But together with the other methods and techniques in the full recommendation system, it does work quite well. 

