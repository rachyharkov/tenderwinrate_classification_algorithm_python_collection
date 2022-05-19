from cProfile import run
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.model_selection import cross_val_score
import os
# import standardScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from flask import Flask, request, jsonify

def initialization(new_x, filename):
    cols = ['harga', 'partner', 'competitor', 'winrate']

    #define app
    app = Flask(__name__, instance_relative_config=True)
    # read csv from instance\sample_data
    dataset = pd.read_csv(os.path.join(app.instance_path, 'sample_data', filename), sep=";", header=None, names=cols)

    x = dataset.iloc[:, :3].values
    y = dataset.iloc[:, -1].values

    for i in range(len(x)):
        for j in range(len(x[i])):
            # trim space
            x[i][j] = x[i][j].strip().replace(" ", "")
            x[i][j] = x[i][j].lower()
            
    winlosscategory = np.unique(y)
    # get the unique values of categorical data
    rangeUnique = np.unique(x)

    newRangeUnique = []
    # reposition rangeUnique, move to index 0 if value is Optimis, index 1 if values is medium, index 2 if value is moderate
    for i in range(len(rangeUnique)):

        if rangeUnique[i] == "moderate":
            newRangeUnique.insert(0, rangeUnique[i])
        if rangeUnique[i] == "medium":
            newRangeUnique.insert(1, rangeUnique[i])
        if rangeUnique[i] == "optimis":
            newRangeUnique.insert(2, rangeUnique[i])
            
    # then change x based on the unique values
    for i in range(len(x)):
        for j in range(len(x[i])):
            # detect if the value is in the range of the unique values
            if x[i][j] in newRangeUnique:
                x[i][j] = newRangeUnique.index(x[i][j])
    
    # convert y to binary
    y = np.where(y == "Win", 1, 0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0) #70% train data, 30% test data


    # feature scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(x_train,y_train)

    #Predict the response for test dataset
    # y_pred = clf.predict(x_test)
    y_pred = clf.predict(x_test)
    score = clf.score(x_test, y_test) #Accuracy of Decision Tree classifier on test set

    # predict accuracy
    new_y_pred = clf.predict(new_x)
    winLossEvaluation = winlosscategory[new_y_pred[0]]

    # probability of win in percentage
    probabilty = clf.predict_proba(new_x)[0][new_y_pred[0]] * 100

    return {
        "status": "success",
        "message": "Accuracy of Decision Tree classifier on test set: " + str(score),
        "evaluation": winLossEvaluation,
        "probability": str(probabilty) + "%"
    }