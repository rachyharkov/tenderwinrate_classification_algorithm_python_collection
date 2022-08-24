import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, learning_curve, train_test_split # Import train_test_split function
from ..graph_creator import generate_chart_compare_accuracy_value, generate_graph_confusion_matrix, generate_chart_performance_comparison, generate_confusion_matrix_each_model_in_gridspec, generate_roc_curve_each_model

import os


# import standardScaler
from sklearn.preprocessing import StandardScaler
from flask import Flask

from ..preprocessing import preprocessing_data

def initialization(new_X, filename):
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
    
    # print(X)
    # split for train purpose
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) #70% train data, 30% test data


    # feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clfLR = LogisticRegression(penalty='l2',solver='newton-cg')
    clfDT = DecisionTreeClassifier(criterion='gini', max_depth=5)
    #max_depth=6, criterion='entropy'
    clfRF = RandomForestClassifier(max_features=, n_estimators=500)



    # parameterlr = {
    #     'penalty': ['l1', 'l2'],
    #     'C': [0.01, 0.1, 1, 10, 100, 1000],
    #     'solver': ['newton-cg', 'lbfgs', 'liblinear']
    # }

    # parameterdt = {
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': [4, 5, 6, 7, 8],
    # }

    # parameterrf = {
    #     'n_estimators': [200, 500],
    #     'max_features': ['sqrt', 'log2'],
    # }

    # cvFold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # # grid search for LR
    # gridlr = GridSearchCV(estimator=clfLR, param_grid=parameterlr, refit=True, verbose=3, n_jobs=-1, scoring='accuracy', cv=cvFold)
    # gridlr.fit(X_train, y_train)
    
    # # grid search for DT
    # griddt = GridSearchCV(estimator=clfDT, param_grid=parameterdt, refit=True, verbose=3, n_jobs=-1, scoring='accuracy', cv=cvFold)
    # griddt.fit(X_train, y_train)
    

    # # grid search for RF
    # gridrf = GridSearchCV(estimator=clfRF, param_grid=parameterrf, refit=True, verbose=0, n_jobs=-1, scoring='accuracy', cv=cvFold)
    # gridrf.fit(X_train, y_train)

    # print('Best LR Parameter :' + str(gridlr.best_params_))
    # print('Best Score' + str(gridlr.best_score_))
    # print('Best DT Parameter :' + str(griddt.best_params_))
    # print('Best Score' + str(griddt.best_score_))
    # print('Best RF Parameter :' + str(gridrf.best_params_))
    # print('Best Score' + str(gridrf.best_score_))

    # Latih model
    clfLR.fit(X_train,y_train)
    clfDT.fit(X_train,y_train)
    clfRF.fit(X_train,y_train)

    y_predLR = clfLR.predict(X_test)
    y_predDT = clfDT.predict(X_test)
    y_predRF = clfRF.predict(X_test)

    # predict using train data
    y_predLR_train = clfLR.predict(X_train)
    y_predDT_train = clfDT.predict(X_train)
    y_predRF_train = clfRF.predict(X_train)

    print(y_predLR_train)
    print(y_predDT_train)
    print(y_predRF_train)

    print('-----------------------------------------------------')
    print("Hasil Prediksi: ")

    # print y_pred based on label

    print("LR: ", y_predLR)
    print("DT: ", y_predDT)
    print("RF: ", y_predRF)

    # accuracy using test data
    print('-----------------------------------------------------')
    print("Akurasi: ")
    print("LR: ", clfLR.score(X_test,y_test))
    print("DT: ", clfDT.score(X_test,y_test))
    print("RF: ", clfRF.score(X_test,y_test))

    traindatascorearray = []
    testdatascorearray = []
    
    # fill the array with the accuracy score test data
    traindatascorearray.append(clfLR.score(X_train,y_train))
    traindatascorearray.append(clfDT.score(X_train,y_train))
    traindatascorearray.append(clfRF.score(X_train,y_train))

    # fill the array with the accuracy score test data
    testdatascorearray.append(clfLR.score(X_test,y_test))
    testdatascorearray.append(clfDT.score(X_test,y_test))
    testdatascorearray.append(clfRF.score(X_test,y_test))


    lr = []
    dt = []
    rf = []

    # get confusion matrix of each model
    tnlr, fplr, fnlr, tplr = confusion_matrix(y_test, y_predLR).ravel()
    tndt, fpdt, fndt, tpdt = confusion_matrix(y_test, y_predDT).ravel()
    tnrf, fprf, fnrf, tprf = confusion_matrix(y_test, y_predRF).ravel()


    #accuracy
    accuracylr = (tplr + tnlr) / (tplr + tnlr + fplr + fnlr)
    accuracydt = (tpdt + tndt) / (tpdt + tndt + fpdt + fndt)
    accuracyrf = (tprf + tnrf) / (tprf + tnrf + fprf + fnrf)

    # misclassification error
    misclassificationerrorlr = 1 - accuracylr
    misclassificationerrordt = 1 - accuracydt
    misclassificationerrorrf = 1 - accuracyrf

    #precision
    precisionlr = tplr / (tplr + fplr)
    precisiondt = tpdt / (tpdt + fpdt)
    precisionrf = tprf / (tprf + fprf)

    #sensitivity(recall)
    sensitivitylr = tplr / (tplr + fnlr)
    sensitivitydt = tpdt / (tpdt + fndt)
    sensitivityrf = tprf / (tprf + fnrf)

    # specificity()
    specificitylr = tnlr / (tnlr + fplr)
    specificitydt = tndt / (tndt + fpdt)
    specificityrf = tnrf / (tnrf + fprf)

    #f1 score
    f1scorelr = 2 * ((precisionlr * sensitivitylr) / (precisionlr + sensitivitylr))
    f1scoredt = 2 * ((precisiondt * sensitivitydt) / (precisiondt + sensitivitydt))
    f1scorerf = 2 * ((precisionrf * sensitivityrf) / (precisionrf + sensitivityrf))

    # append the value to the list
    lr.append(accuracylr)
    lr.append(misclassificationerrorlr)
    lr.append(precisionlr)
    lr.append(sensitivitylr)
    lr.append(specificitylr)
    lr.append(f1scorelr)

    dt.append(accuracydt)
    dt.append(misclassificationerrordt)
    dt.append(precisiondt)
    dt.append(sensitivitydt)
    dt.append(specificitydt)
    dt.append(f1scoredt)

    rf.append(accuracyrf)
    rf.append(misclassificationerrorrf)
    rf.append(precisionrf)
    rf.append(sensitivityrf)
    rf.append(specificityrf)
    rf.append(f1scorerf)




    # classification report
    print("Classification Report: ")
    print("LR: ", classification_report(y_test, y_predLR))
    print("DT: ", classification_report(y_test, y_predDT))
    print("RF: ", classification_report(y_test, y_predRF))

    # generate_graph_confusion_matrix("LR", y_test, y_predLR, app)
    # generate_graph_confusion_matrix("DT", y_test, y_predDT, app)
    # generate_graph_confusion_matrix("RF", y_test, y_predRF, app)


    # plot sensitivity using train data
    generate_chart_performance_comparison(lr, dt, rf)

     # plot accuracy using train data
    generate_chart_compare_accuracy_value(traindatascorearray, testdatascorearray)
    # plot confusion matrix of each model
    generate_confusion_matrix_each_model_in_gridspec(tnlr, fplr, fnlr, tplr, tndt, fpdt, fndt, tpdt, tnrf, fprf, fnrf, tprf)

    # plot roc curve of each model
    generate_roc_curve_each_model(y_test, y_predLR, y_predDT, y_predRF)


    