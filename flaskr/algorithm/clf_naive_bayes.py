import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from flask import Flask
from ..preprocessing import preprocessing_data
from ..graph_creator import generate_graph_confusion_matrix, generate_graph_tree_path, generate_graph_tree_complete
import os

def initialization(new_X, filename):

    cols = ['harga', 'partner', 'competitor', 'winrate']

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # feature scaling
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #Create a Gaussian Classifier
    clf = GaussianNB()

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    test_data_score = clf.score(X_test, y_test)
    train_data_score = clf.score(X_train, y_train)

    new_X_test = sc.transform(new_X)

    new_y_pred = clf.predict(new_X_test)
    winLossEvaluation = winlosscategory[new_y_pred[0]]
   

    urlcm, cmdetail = generate_graph_confusion_matrix('nb_CM' + filename + '.png',y_test, y_pred,app)

    # probability
    new_prob = clf.predict_proba(new_X_test)
    pathtree = 'http://localhost:5000/graph/?name=na'
    
    return {
        "status": "success",
        "algorithm_name": "Naive Bayes",
        "evaluation": winLossEvaluation,
        "probability": {
            "lose": str(new_prob[0][0] * 100),
            "win": str(new_prob[0][1] * 100)
        },
        "graph": {
            "confusion_matrix": {
                "picture": urlcm,
                "detail": '<b>Berdasarkan dataset yang anda upload</b>, Naive Bayes menghasilkan <i><b>' + str(cmdetail[0][0]) + '</b></i> data tender diprediksi tidak akan dimenangi dan data asli menyatakan demikian | <i><b>' + str(cmdetail[0][1]) + '</b></i> data diprediksi menang walaupun data asli mengatakan kalah| <i><b>' + str(cmdetail[1][0]) + ' data</b></i> diprediksi kalah walaupun data asli menyatakan menang | <i><b>' + str(cmdetail[1][1]) + '</b></i> data diprediksi menang dan data asli menyatakan demikian.'
            },
            "tree" : {
                "picture": pathtree,
                "detail": "BLABLABLA TREEEEE"
            }
        },
        "accuracy": {
            "test": str(test_data_score),
            "train": str(train_data_score)
        },
        "precision": {
            "test": str(precision_score(y_test, y_pred))
        },
        "recall": {
            "test": str(recall_score(y_test, y_pred)),
        },
        "f1_score": {
            "test": str(f1_score(y_test, y_pred)),
        },
        "report": {
            "test": str(classification_report(y_test, y_pred, target_names=['lose', 'win']))
        }
    }

# print(initialization(np.array([[0,1,0]]), "training_data.csv"))