
import os
import pandas as pd
from flask import Flask
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report 

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ..preprocessing import preprocessing_data
from ..graph_creator import generate_graph_confusion_matrix, generate_graph_tree_path, generate_graph_tree_complete

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
    winorlosslabel = preprocessed[2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # feature scaling
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # n estimator adalah banyak pohon yang ingin dibuat pada hutan , nilai harus integer
    clf = RandomForestClassifier(criterion='gini', n_estimators=200, random_state=42, max_depth=6, max_features='auto')
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_data_score = clf.score(X_test, y_test)
    train_data_score = clf.score(X_train, y_train)

    print('Data baru sebelum di transform :', new_x)
    new_X_test = sc.transform(new_x)
    print('Data baru setelah di transform :', new_X_test)
    # predict the result
    new_y_pred = clf.predict(new_X_test)

    feat_importance = clf.feature_importances_
    print('Feature Importance :' + str(feat_importance))

     # plot feature importance
    import matplotlib.pyplot as plt
    plt.bar(['harga', 'partner','competitor'], feat_importance)
    plt.show()  
    
    new_prob = clf.predict_proba(new_X_test)
    winLossEvaluation = winorlosslabel[new_y_pred[0]]

    urlpathtree = generate_graph_tree_path(clf.estimators_[0], nama_kolom, ['Lose', 'Win'], new_X_test, 'rf_ptree' + filename + '.png', app)
    urlcompletetree = generate_graph_tree_complete(clf.estimators_[0], nama_kolom, ['Lose', 'Win'], 'rf_tree' + filename + '.png', app)
    urlcm, cmdetail = generate_graph_confusion_matrix('rf_CM' + filename + '.png',y_test, y_pred,app)
    
    return {
        "status": "success",
        "algorithm_name": "Random Forest",
        "evaluation": winLossEvaluation,
        "probability": {
            "lose": str(new_prob[0][0] * 100),
            "win": str(new_prob[0][1] * 100)
        },
        "graph": {
            "confusion_matrix": {
                "picture": urlcm,
                "detail": '<b>Berdasarkan dataset yang anda upload</b>, Random Forest menghasilkan <i><b>' + str(cmdetail[0][0]) + '</b></i> data tender diprediksi tidak akan dimenangi dan data asli menyatakan demikian | <i><b>' + str(cmdetail[0][1]) + '</b></i> data diprediksi menang walaupun data asli mengatakan kalah| <i><b>' + str(cmdetail[1][0]) + ' data</b></i> diprediksi kalah walaupun data asli menyatakan menang | <i><b>' + str(cmdetail[1][1]) + '</b></i> data diprediksi menang dan data asli menyatakan demikian.'
            },
            "tree" : {
                "picture": urlcompletetree,
                "detail": "BLABLABLA TREEEEE"
            },
            "treepath" : {
                "picture": urlpathtree,
                "detail": "BLABLABLA TREEPATH"
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
            "test": str(classification_report(y_test, y_pred, target_names=winorlosslabel)),
        }
    }