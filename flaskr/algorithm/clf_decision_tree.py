from cProfile import run
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
import os
# import standardScaler
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from flask import Flask
from matplotlib import pyplot as plt
import pydot



def initialization(new_x, filename):
    cols = ['harga', 'partner', 'competitor', 'winrate']

    colny = ['harga', 'partner', 'competitor']

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

    # split for train purpose
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0) #70% train data, 30% test data


    # feature scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    clf = DecisionTreeClassifier(random_state=0)

    # Train Decision Tree Classifer
    clf = clf.fit(x_train,y_train)

    #Predict the response for test dataset
    # y_pred = clf.predict(x_test)
    y_pred = clf.predict(x_test)
    score = clf.score(x_test, y_test) #Accuracy of Decision Tree classifier on test set

    # predict new x
    new_x_test = sc.transform(new_x)
    # predict accuracy
    new_y_pred = clf.predict(new_x_test)
    winLossEvaluation = winlosscategory[new_y_pred[0]]

    # probability of win in percentage
    probabilty = clf.predict_proba(new_x_test)[0][new_y_pred[0]] * 100

    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(clf, 
                   feature_names=colny,  
                   class_names=winlosscategory,
                   filled=True)
    fig.savefig(os.path.join(app.instance_path, 'graph_data', 'decision_tree' + filename + '.png'))
    
    # get path of graph
    path = 'graph_data/decision_tree' + filename + '.png'
    
    return {
        "status": "success",
        "algorithm_name": "Decision Tree",
        "message": "Accuracy of Decision Tree classifier on test set: " + str(score),
        "evaluation": winLossEvaluation,
        "probability": str(probabilty) + "%",
        "graph": path
    }

print(initialization(np.array([[2,1,2]]), "training_data.csv"))