# import gridsearchcv
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedKFold
import pandas as pd
from ..preprocessing import preprocessing_data

def do_parameter_tuning(file):

    dataset = pd.read_csv(file, sep=';', header=None, names=['harga', 'partner', 'competitor', 'result'])

    X = dataset.iloc[:, :3].values
    y = dataset.iloc[:, -1].values

    preprocessed = preprocessing_data(X,y)

    X = preprocessed[0]
    y = preprocessed[1]
    winorlosslabel = preprocessed[2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    clfLR = LogisticRegression()
    clfDT = DecisionTreeClassifier()
    clfRF = RandomForestClassifier()

    parameterlr = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'solver': ['liblinear', 'newton-cg', 'lbfgs'],
    }

    parameterdt = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [4, 5, 6, 7, 8]
    }

    parameterrf = {
        'n_estimators': [200, 500],
        'criterion': ['gini', 'entropy'],
        'max_depth': [4, 5, 6, 7, 8]
    }

    cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    gridlr = GridSearchCV(estimator=clfLR, refit=True, verbose=True, param_grid=parameterlr, cv=cvFold, n_jobs=-1, scoring='accuracy')
    gridlr.fit(X_train, y_train)
    griddt = GridSearchCV(estimator=clfDT, refit=True, verbose=True, param_grid=parameterdt, cv=cvFold, n_jobs=-1, scoring='accuracy')
    griddt.fit(X_train, y_train)
    gridrf = GridSearchCV(estimator=clfRF, refit=True, verbose=True, param_grid=parameterrf, cv=cvFold, n_jobs=-1, scoring='accuracy')
    gridrf.fit(X_train, y_train)


    # print best parameter after tuning 
    print(gridlr.best_params_)
    print(griddt.best_params_)
    print(gridrf.best_params_)