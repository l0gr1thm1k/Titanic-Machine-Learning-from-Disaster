CONTEST INFO
============

This is the source code of the submission by l0gr1thm1k for the Kaggle introduction contest of predicting which passengers will survive the sinking of the Titanic.
The code has brought me to the top 25th percentile (1787/6977, with 9 entries), as of 11/05/2017.

Current high scoring submission: 0.79426. For context, as of 11/05/2017 a prediction accuracy improvement of 1% will increase my standing in the competition by 1178 places. 

Details of the contest:

https://www.kaggle.com/c/titanic

LIBRARY
--------
I used sklearn, numpy and pandas to complete this conteset.

MODELS TRAINED
---------------

1. The winning algorithm so far is SVM with an RBF kernel. 
2. Other classifiers have not yet scored as well, but with carefully processing the data may yield better results. 
classifier algorithms and their highest scoring submissions: 
   * SVM with RBF kernel - 0.79426
   * Logistic Regression - 0.76077 
   * K-Nearest Neighbors - 0.7461 
   * Random Forest - 0.7461
   * Gaussian Naïve Bayes - 0.71292

EVALUATION
-----------

The metric used on the contest was prediction accuracy. 


OTHER IDEAS
----------------------------------

So far most the work in improving prediction scores has been by utilizing different classifiers. There is potential for large increases in prediction accuracy by teasing out relationships between various fields in the data set. For example, likelihood of survival is affected by whether or not the passenger had family on board, and if the family members also survived [Do Families Survive Together?](https://www.kaggle.com/philippsp/increase-your-model-accuracy-with-this-feature).
