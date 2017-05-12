# import the libraries
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler


def preprocess(fname):
    dataset = pd.read_csv(fname)
    test_set = False
    if len(dataset.columns.values) == 11:
        test_set = True
    if test_set:
        
        X = dataset.iloc[:, [1, 3, 4, 5, 6, 8, 10]].values
        ground_truth_data = pd.read_csv("gender_submission.csv")
        y = ground_truth_data.iloc[:, 1].values
    else:
        X = dataset.iloc[:, [2, 4, 5, 6, 7, 9, 11]].values
        y = dataset.iloc[:, 1].values

    label_encoder_sex = LabelEncoder()
    X[:, 1] = label_encoder_sex.fit_transform(X[:, 1])
    X[:, -1] = np.array([np.int(ord(x)) if type(x) == str else x for x in X[:, -1]])
    

    
    imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
    imputer.fit(X[:, :])
    X[:, :] = imputer.transform(X[:, :])

    one_hot_encoder = OneHotEncoder(categorical_features=[0], dtype=np.int)      
    X = one_hot_encoder.fit_transform(X).toarray()

    sc = StandardScaler()
    X = sc.fit_transform(X)
    
    return X, y

def make_submission(y_pred):
    ids = pd.read_csv("gender_submission.csv").iloc[:, 0].values
    submission = zip(ids, y_pred)
    submission_file = open("submission.csv", "w")
    submission_file.write("PassengerID,Survived\n")
    for element in submission:
        submission_file.write(",".join([str(x) for x in element]) + "\n")
    submission_file.close()

if __name__ == "__main__":
    X_train, y_train = preprocess("train.csv")
    X_test, y_test = preprocess("test.csv")
    
    # create a classifier and fit it to the data
    
    # logistic regression
    # from sklearn.linear_model import LogisticRegression
    # classifier = LogisticRegression(random_state=0)
    
    # SVM kernel
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf', random_state=0, C=1.0)
    
    # K-Nearest Neighbors
    # from sklearn.neighbors import KNeighborsClassifier
    # classifier = KNeighborsClassifier(n_neighbors=5)
    
    # Naive Bayes
    # from sklearn.naive_bayes import GaussianNB
    # classifier = GaussianNB()
    
    # Decision Tree
    # from sklearn.ensemble import RandomForestClassifier
    # classifier = RandomForestClassifier(n_estimators=500)

    # make predictions on the test set and a submission file
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    make_submission(y_pred)
    
    # make a confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)