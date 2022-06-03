import numpy as np
import os
import pandas as pd
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    # feature scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # n estimator adalah banyak pohon yang ingin dibuat pada hutan , nilai harus integer
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    score = clf.score(x_test, y_test) #Accuracy of Random Forest classifier on test set

    new_x_test = sc.transform(new_x)
    # predict the result
    new_y_pred = clf.predict(new_x_test)
    probability = clf.predict_proba(new_x_test)[0][new_y_pred[0]] * 100
    winLossEvaluation = winlosscategory[new_y_pred[0]]

    return {
        "status": "success",
        "filename": filename,
        "algorithm_name": "Random Forest",
        "message": "Accuracy of Random Forest Classifier on test set: " + str(score) ,
        "evaluation": winLossEvaluation,
        "probability": str(probability) + "%"
    }

print(initialization(np.array([[2,1,2]]), "training_data.csv"))