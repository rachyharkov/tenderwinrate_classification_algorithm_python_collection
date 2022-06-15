import numpy as np
import os
import pandas as pd
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocessing_data(X,y):

    for i in range(len(X)):
        for j in range(len(X[i])):
            # trim space
            X[i][j] = X[i][j].strip().replace(" ", "")
            X[i][j] = X[i][j].lower()

    winlosscategory = np.unique(y)
    # get the unique values of categorical data
    rangeUnique = np.unique(X)

    newRangeUnique = []
    # reposition rangeUnique, move to index 0 if value is Optimis, index 1 if values is medium, index 2 if value is moderate
    for i in range(len(rangeUnique)):

        if rangeUnique[i] == "moderate":
            newRangeUnique.insert(0, rangeUnique[i])
        if rangeUnique[i] == "medium":
            newRangeUnique.insert(1, rangeUnique[i])
        if rangeUnique[i] == "optimis":
            newRangeUnique.insert(2, rangeUnique[i])



    print(newRangeUnique)
    # then change x based on the unique values
    for i in range(len(X)):
        for j in range(len(X[i])):
            # detect if the value is in the range of the unique values
            if X[i][j] in newRangeUnique:
                X[i][j] = newRangeUnique.index(X[i][j])


    # convert y to binary
    y = np.where(y == "Win", 1, 0)

    return X,y,winlosscategory

def initialization(new_x, filename):

    cols = ['harga', 'partner', 'competitor', 'winrate']

    nama_kolom = ['harga', 'partner', 'competitor']

    #define app
    app = Flask(__name__, instance_relative_config=True)
    # read csv from instance\sample_data
    dataset = pd.read_csv(os.path.join(app.instance_path, 'sample_data', filename), sep=";", header=None, names=cols)

    X = dataset.iloc[:, :3].values
    y = dataset.iloc[:, -1].values

    preprocessed = preprocessing_data(X,y)

    X = preprocessed[0]
    y = preprocessed[1]
    winlosscategory = preprocessed[2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # n estimator adalah banyak pohon yang ingin dibuat pada hutan , nilai harus integer
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    test_data_score = clf.score(X_test, y_test)
    train_data_score = clf.score(X_train, y_train)

    new_X_test = sc.transform(new_x)
    # predict the result
    new_y_pred = clf.predict(new_X_test)
    new_prob = clf.predict_proba(new_X_test)
    winLossEvaluation = winlosscategory[new_y_pred[0]]

    return {
        "status": "success",
        "filename": filename,
        "algorithm_name": "Random Forest",
        "message": "Accuracy on test set: " + str(test_data_score) + "\n Accuracy on train set: " + str(train_data_score),
        "evaluation": winLossEvaluation,
        "probability": {
            "lose": str(new_prob[0][0] * 100),
            "win": str(new_prob[0][1] * 100)
        }
    }

print(initialization(np.array([[0,1,0]]), "training_data.csv"))