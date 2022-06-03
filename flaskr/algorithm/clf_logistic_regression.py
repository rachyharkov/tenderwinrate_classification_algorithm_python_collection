import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify
import os

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



    print(newRangeUnique)
    # then change x based on the unique values
    for i in range(len(x)):
        for j in range(len(x[i])):
            # detect if the value is in the range of the unique values
            if x[i][j] in newRangeUnique:
                x[i][j] = newRangeUnique.index(x[i][j])


    # convert y to binary
    y = np.where(y == "Win", 1, 0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


    # feature scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    logisticReg = LogisticRegression()
    logisticReg.fit(x_train, y_train)
    logisticReg.predict(x_test[0].reshape(1, -1))
    logisticReg.predict(x_test[0:10])
    predictions = logisticReg.predict(x_test)
    prob = logisticReg.predict_proba(x_test)[0][1]


    new_x_test = sc.transform(new_x)

    newprediction = logisticReg.predict(new_x_test)
    # probability of the result
    new_prob = logisticReg.predict_proba(new_x_test)[0][1]

    return {
        "status": "success",
        "algorithm_name": "Logistic Regression",
        "message": "Accuracy of logistic regression classifier on test set: " + str(logisticReg.score(x_test, y_test)),
        "evaluation": winlosscategory[newprediction[0]],
        "probability": str(new_prob * 100) + "%"
    }