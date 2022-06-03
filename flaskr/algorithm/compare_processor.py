# Compare Algorithms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from flask import Flask, request, jsonify
import os

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

    # convert X as scale data 

    return X,y,winlosscategory

def initialization(new_x, filename):

    cols = ['harga', 'partner', 'competitor', 'winrate']

    #define app
    app = Flask(__name__, instance_relative_config=True)
    # read csv from instance\sample_data
    dataset = pd.read_csv(os.path.join(app.instance_path, 'sample_data', filename), sep=";", header=None, names=cols)

    X = dataset.iloc[:, :3].values
    Y = dataset.iloc[:, -1].values

    preprocessed = preprocessing_data(X,Y)
    X = preprocessed[0]
    Y = preprocessed[1]
    # prepare configuration for cross validation test harness
    seed = 7
    # prepare models
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

print(initialization(np.array([[0,1,0]]), "training_data.csv"))