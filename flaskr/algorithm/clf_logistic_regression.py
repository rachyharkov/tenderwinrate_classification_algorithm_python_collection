import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
from flask import Flask
from ..preprocessing import preprocessing_data
from ..graph_creator import generate_graph_confusion_matrix, generate_graph_curve_probability
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

    # feature scaling using minmaxscaler

    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    clf = LogisticRegression(C=1, penalty='l2',solver='newton-cg')
    clf = clf.fit(X_train,y_train)

    #Predict the response using test dataset
    y_pred = clf.predict(X_test)

    y_pred_proba = clf.predict_proba(X_test)
    

    test_data_score = clf.score(X_test, y_test)
    train_data_score = clf.score(X_train, y_train)

    
    # predict new X [0,1,0]
    new_X_test = sc.transform(new_X)

    new_y_pred = clf.predict(new_X_test)
    new_y_pred_proba = clf.predict_proba(new_X_test)

    winLossEvaluation = winlosscategory[new_y_pred[0]]
    
    urlcm, cmdetail = generate_graph_confusion_matrix('lr_CM' + filename + '.png',y_test, y_pred,app)
    curveproburl = generate_graph_curve_probability(y_pred_proba, 'lr_curveprob x' + filename + '.png', app)

    # probability
    new_prob = clf.predict_proba(new_X_test)

    return {
        "status": "success",
        "algorithm_name": "Logistic Regression",
        "evaluation": winLossEvaluation,
        "probability": {
            "lose": str(new_prob[0][0] * 100),
            "win": str(new_prob[0][1] * 100)
        },
        "graph": {
            "confusion_matrix": {
                "picture": urlcm,
                "detail": '<b>Berdasarkan dataset yang anda upload</b>, Logistic Regression menghasilkan <i><b>' + str(cmdetail[0][0]) + '</b></i> data tender diprediksi tidak akan dimenangi dan data asli menyatakan demikian | <i><b>' + str(cmdetail[0][1]) + '</b></i> data diprediksi menang walaupun data asli mengatakan kalah| <i><b>' + str(cmdetail[1][0]) + ' data</b></i> diprediksi kalah walaupun data asli menyatakan menang | <i><b>' + str(cmdetail[1][1]) + '</b></i> data diprediksi menang dan data asli menyatakan demikian.'
            },
            "curve_probability": {
                "picture": curveproburl,
                "detail": 'Probability of Win and Lose'
            },
            "tree" : {
                "picture": 'na',
                "detail": "BLABLABLA TREEEEE"
            },
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
            "test": str(classification_report(y_test, y_pred, target_names=['Lose', 'Win']))
        }
    }