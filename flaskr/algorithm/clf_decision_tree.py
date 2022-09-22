import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.tree import DecisionTreeClassifier, export_text # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from ..preprocessing import preprocessing_data
from ..graph_creator import generate_graph_confusion_matrix, generate_graph_tree_path, generate_graph_tree_complete
import os
from sklearn.preprocessing import MinMaxScaler
from flask import Flask

def initialization(new_X, filename):
    cols = ['harga', 'partner', 'competitor', 'winrate']

    nama_kolom = ['harga', 'partner', 'competitor']

    #define app
    app = Flask(__name__, instance_relative_config=True)
    # read csv from instance\sample_data
    dataset = pd.read_csv(os.path.join(app.instance_path, 'sample_data', filename), sep=";", header=None, names=cols)

    X = dataset.iloc[:, :3].values
    y = dataset.iloc[:, -1].values

    print('Dataset :')
    print(X)
    print(y)

    preprocessed = preprocessing_data(X,y)

    X = preprocessed[0]
    y = preprocessed[1]
    winorlosslabel = preprocessed[2]

    # print(X)
    # split for train purpose
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) #70% train data, 30% test data


    # feature scaling
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = DecisionTreeClassifier(criterion='gini', max_depth=5)
    
    # Latih model
    clf = clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    
    test_data_score = clf.score(X_test, y_test)
    train_data_score = clf.score(X_train, y_train)
    # predict new X

    new_X_test = sc.transform(new_X)
    
    new_y_pred = clf.predict(new_X_test)

    winLossEvaluation = winorlosslabel[new_y_pred[0]]

    urlpathtree = generate_graph_tree_path(clf, nama_kolom, ['Lose', 'Win'], new_X_test, 'dtCART_ptree' + filename + '.png', app)
    urlcompletetree = generate_graph_tree_complete(clf, nama_kolom, ['Lose', 'Win'], 'dtCART_tree' + filename + '.png', app)
    urlcm, cmdetail = generate_graph_confusion_matrix('dtCART_CM' + filename + '.png',y_test, y_pred,app)

    # probability
    new_prob = clf.predict_proba(new_X_test)

    return {
        "status": "success",
        "algorithm_name": "Decision Tree",
        "evaluation": winLossEvaluation,
        "probability": {
            "lose": str(new_prob[0][0] * 100),
            "win": str(new_prob[0][1] * 100)
        },
        "graph": {
            "confusion_matrix": {
                "picture": urlcm,
                "detail": '<b>Berdasarkan dataset yang anda upload</b>, Decision tree menghasilkan <i><b>' + str(cmdetail[0][0]) + '</b></i> data tender diprediksi tidak akan dimenangi dan data asli menyatakan demikian | <i><b>' + str(cmdetail[0][1]) + '</b></i> data diprediksi menang walaupun data asli mengatakan kalah| <i><b>' + str(cmdetail[1][0]) + ' data</b></i> diprediksi kalah walaupun data asli menyatakan menang | <i><b>' + str(cmdetail[1][1]) + '</b></i> data diprediksi menang dan data asli menyatakan demikian.'
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
        "report" : {
            "test": str(classification_report(y_test, y_pred, target_names=winorlosslabel))
        }
    }

# print(initialization(np.array([[0,1,0]]), "training_data.csv"))