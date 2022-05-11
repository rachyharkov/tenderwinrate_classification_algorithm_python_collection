from cProfile import run
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.model_selection import cross_val_score
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation



def run_model(new_x):
    print('Processing data...')

    dataset = pd.read_csv("training_data.csv", header=None, sep=';')

    x = dataset.iloc[:, :3].values
    y = dataset.iloc[:, -1].values

    # append new data to the dataset

    print(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20) # 80% training and 20% test

    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(x_train,y_train)

    #Predict the response for test dataset
    # y_pred = clf.predict(x_test)

    
    y_pred = clf.predict(x_test)
    # predict accuracy
    winLossEvaluation = winlossunique[y_pred[0]]

    # accuracy = np.mean(cross_val_score(clf, x_test, y_pred, scoring='accuracy')) * 100
    # string = "Input Test for : "

    y_pred = clf.predict(new_x)
    # for x in x_test:
    #     string += str(x) + ' (' + str(rangeUnique[x]) + ')'
    print(metrics.classification_report(y_test, y_pred))

    # print(string)
    # print("Accuracy: {}%".format(accuracy))
    print("Win Loss Evaluation: ", winLossEvaluation)

    exit()

def initialization():
    print("-------------- DECISION TREE PREDICTION - UNDER DEVELOPMENT --------------")
    print("Please, enter the following data based on range:")
    print("0 = Medium")
    print("1 = Moderate")
    print("2 = Optimis")
    print("-----------------------------------------------------")
    harga_val = input("Harga : ")
    partner_val = input("Partner : ")
    competitor_val = input("Competitor : ")

    arrayofInput = [harga_val, partner_val, competitor_val]

    new_x_test = [arrayofInput]
    run_model(new_x_test)

initialization()