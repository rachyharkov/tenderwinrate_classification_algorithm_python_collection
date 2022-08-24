from flask import Flask
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pydotplus
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn import tree
import os

app = Flask(__name__, instance_relative_config=True)

def generate_graph_curve_probability(y_pred_proba, filename, app):
    figax = plt.figure(figsize=(10,3))
    gs = gridspec.GridSpec(1, 2)
    ax1 = plt.subplot(gs[0, 0])
    # plot graph curve comparison of y_pred_proba and y_test
    ax1.plot(y_pred_proba[:,1], 'b')
    ax1.set_xlabel('sample')
    ax1.set_ylabel('result')
    ax1.legend()
    ax1.set_title('Win Prediction Probability')

    ax2 = plt.subplot(gs[0, 1])
    # plot graph curve comparison of y_pred_proba and y_test
    ax2.plot(y_pred_proba[:,0], 'r')
    ax2.set_xlabel('sample')
    ax2.set_ylabel('result')
    ax2.legend()
    ax2.set_title('Lose Prediction Probability')
    # plt.show()

    # show plot
    # save plot
    figax.savefig(os.path.join(app.instance_path, 'graph_data', filename))
    curveproburl = 'http://localhost:5000/graph/?name=' + filename
    return curveproburl
    

def generate_graph_confusion_matrix(filename, y_test, y_pred, app):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cnf_matrix.ravel()

    #accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # misclassification error
    misclassificationerror = 1 - accuracy
    #precision
    precision = tp / (tp + fp)
    #sensitivity(recall)
    sensitivity = tp / (tp + fn)
    # specificity
    specificity = tn / (tn + fp)
    #f1 score
    f1score = 2 * ((precision * sensitivity) / (precision + sensitivity))
    
    labels = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    # create heatmap
    _ = sns.heatmap([[fp, tn], [tp, fn]], annot = True, fmt = 'd', xticklabels = [1,0], yticklabels = [0,1])
    ax.xaxis.set_label_position("top")
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')

    fig.savefig(os.path.join(app.instance_path, 'graph_data', filename))
    confusionmatrixurl = 'http://localhost:5000/graph/?name=' + filename
    return confusionmatrixurl, cnf_matrix

def generate_confusion_matrix_each_model_in_gridspec(tnlr, fplr, fnlr, tplr, tndt, fpdt, fndt, tpdt, tnrf, fprf, fnrf, tprf):
    
    sns.set(font_scale=1.5)
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(12,10), facecolor ='white')

    ax1 = plt.subplot(gs[0, 0])
    # create heatmap
    # set label of each cell
    _ = sns.heatmap([[fplr, tnlr], [tplr, fnlr]], annot = True, fmt = 'd', xticklabels = [1,0], yticklabels = [0,1])
    # set the x and y label
    ax1.xaxis.set_label_position("top")
    plt.title('CF LR', y=1.05, fontsize=10)
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    ax2 = plt.subplot(gs[0, 1])
    # create heatmap
    _ = sns.heatmap([[fpdt, tndt], [tpdt, fndt]], annot = True, fmt = 'd', xticklabels = [1,0], yticklabels = [0,1])
    ax2.xaxis.set_label_position("top")
    plt.title('CF DT', y=1.05, fontsize=10)
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')
    ax3 = plt.subplot(gs[1, 0])
    # create heatmap
    _ = sns.heatmap([[fprf, tnrf], [tprf, fnrf]], annot = True, fmt = 'd', xticklabels = [1,0], yticklabels = [0,1])

    ax3.xaxis.set_label_position("top")
    plt.title('CF RF', y=1.05, fontsize=10)
    plt.ylabel('Aktual')
    plt.xlabel('Prediksi')

    # plot
    # plt.show()
    # save
    fig.savefig(os.path.join(app.instance_path, 'graph_data', 'confusion_matrix_each_model_in_gridspec.png'))
    confusionmatrixurl = 'http://localhost:5000/graph/?name=confusion_matrix_each_model_in_gridspec.png'
    return confusionmatrixurl
    

def generate_graph_tree_complete(clf, feature_names, target_class, filename, app):
    plt.figure(figsize=(30,10), facecolor ='k')

    figTree = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(clf, 
                   feature_names=feature_names,  
                   class_names=target_class,
                   filled=True)
    figTree.savefig(os.path.join(app.instance_path, 'graph_data', filename))
    treecomplete = 'http://localhost:5000/graph/?name=' + filename
    return treecomplete

def generate_graph_tree_path(clf, feature_names, target_class, X, filename, app):
    dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=feature_names,
                                class_names=target_class,
                                filled=True, rounded=True,
                                special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    for node in graph.get_node_list():
        if node.get_attributes().get('label') is None:
            continue
        if 'samples = ' in node.get_attributes()['label']:
            labels = node.get_attributes()['label'].split('<br/>')
            for i, label in enumerate(labels):
                if label.startswith('samples = '):
                    labels[i] = 'samples = 0'
            node.set('label', '<br/>'.join(labels))
            node.set_fillcolor('white')

    decision_paths = clf.decision_path(X)
    for decision_path in decision_paths:
        for n, node_value in enumerate(decision_path.toarray()[0]):
            if node_value == 0:
                continue
            node = graph.get_node(str(n))[0]            
            node.set_fillcolor('green')
            labels = node.get_attributes()['label'].split('<br/>')
            for i, label in enumerate(labels):
                if label.startswith('samples = '):
                    labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)

            node.set('label', '<br/>'.join(labels))

    # save the graph as file to instances/graph_data
    graph.write_png(os.path.join(app.instance_path, 'graph_data', filename))
    completetreeurl = 'http://localhost:5000/graph/?name=' + filename
    return completetreeurl

def generate_chart_compare_accuracy_value(traindatascore, testdatascore):
    # convert to percentage
    traindatascore = [x * 100 for x in traindatascore]
    testdatascore = [x * 100 for x in testdatascore]
    
    barWidth = 0.25
    fig = plt.figure(figsize=(10,10), facecolor ='white')
    
    # Set position of bar on X axis
    br1 = np.arange(len(traindatascore))
    br2 = [x + barWidth for x in br1]


    # Make the plot
    plt.bar(br1, traindatascore, color ='r', width = barWidth,
            edgecolor ='grey', label ='Data Test')
    plt.bar(br2, testdatascore, color ='g', width = barWidth,
            edgecolor ='grey', label ='Data Train')

    
    # place percentage text on top of each bar
    for i, v in enumerate(traindatascore):
        plt.text(x = br1[i] - 0.10, y = v - 0.80, s = str(round(v,2)) + '%', color = 'black', fontsize = 14)
    for i, v in enumerate(testdatascore):
        plt.text(x = br1[i] + 0.14, y = v - 0.80, s = str(round(v,2)) + '%', color = 'black', fontsize = 14) 

    plt.xlabel('Model', fontweight ='bold', fontsize = 15)
    plt.ylabel('Tingkat Akurasi', fontweight ='bold', fontsize = 15)
    
    plt.xticks([r + barWidth for r in range(len(traindatascore))],
            ['Logistic Regression', 'Decision Tree', 'Random Forest'])
    # increase font size of x-axis ticks
    plt.tick_params(axis='x', which='major', labelsize=15)   

    plt.legend()
    # plt.show()
    # save the graph as file to instances/graph_data
    fig.savefig(os.path.join(app.instance_path, 'graph_data', 'compare_accuracy_value.png'))
    compareaccuracyurl = 'http://localhost:5000/graph/?name=compare_accuracy_value.png'
    return compareaccuracyurl

def generate_chart_performance_comparison(lr, dt, rf):
    
    # convert to percentage
    lr = [x * 100 for x in lr]
    dt = [x * 100 for x in dt]
    rf = [x * 100 for x in rf]


    barWidth = 0.27
    fig = plt.figure(figsize=(10,5), facecolor ='white')
    
    # Set position of bar on X axis
    br1 = np.arange(len(lr))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, lr, color ='r', width = barWidth,
            edgecolor ='grey', label ='Logistic Regression')
    plt.bar(br2, dt, color ='g', width = barWidth,
            edgecolor ='grey', label ='Decision Tree')
    plt.bar(br3, rf, color ='b', width = barWidth,
            edgecolor ='grey', label ='Random Forest')
    
    # place percentage text on top of each bar
    for i, v in enumerate(lr):
        plt.text(x = br1[i] - 0.15, y = v - 4.70, s = str(round(v,2)) + '%', color = 'black', fontsize = 10)
    for i, v in enumerate(dt):
        plt.text(x = br2[i] - 0.15, y = v - 0.10, s = str(round(v,2)) + '%', color = 'black', fontsize = 10)
    for i, v in enumerate(rf):
        plt.text(x = br3[i] - 0.13, y = v - 4.70, s = str(round(v,2)) + '%', color = 'black', fontsize = 10)
    
    plt.xticks([r + barWidth for r in range(len(lr))],
            ['Accuracy','Misclassification','Precision', 'Sensitivity \n(Recall)', 'Specificity', 'F1 Score'])
    

    # increase font size of x-axis ticks
    plt.tick_params(axis='x', which='major', labelsize=13)
    plt.legend()
    # put the legend in the best location
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), fancybox=True, shadow=True, ncol=5)
    
    # plt.show()
    # save the graph as file to instances/graph_data
    fig.figure.savefig(os.path.join(app.instance_path, 'graph_data', 'performance_comparison.png'))
    performancecomparisonurl = 'http://localhost:5000/graph/?name=performance_comparison.png'
    return performancecomparisonurl

def generate_roc_curve_each_model(y_test, y_predLR, y_predDT, y_predRF):
    # calculate roc curve for each model
    fprLR, tprLR, thresholdsLR = roc_curve(y_test, y_predLR)
    fprDT, tprDT, thresholdsDT = roc_curve(y_test, y_predDT)
    fprRF, tprRF, thresholdsRF = roc_curve(y_test, y_predRF)
    
    # calculate roc auc for each model
    roc_aucLR = auc(fprLR, tprLR)
    roc_aucDT = auc(fprDT, tprDT)
    roc_aucRF = auc(fprRF, tprRF)
    
    # plot the roc curve for each model
    fig = plt.figure(figsize=(10,10), facecolor ='white')
    plt.plot(fprLR, tprLR, color='red', lw=2, label='Logistic Regression (area = %0.2f)' % roc_aucLR)
    plt.plot(fprDT, tprDT, color='green', lw=2, label='Decision Tree (area = %0.2f)' % roc_aucDT)
    plt.plot(fprRF, tprRF, color='blue', lw=2, label='Random Forest (area = %0.2f)' % roc_aucRF)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight ='bold', fontsize = 15)
    plt.ylabel('True Positive Rate', fontweight ='bold', fontsize = 15)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight ='bold', fontsize = 15)
    plt.legend(loc="lower right")

    # plt.show()
    # save the graph as file to instances/graph_data
    fig.savefig(os.path.join(app.instance_path, 'graph_data', 'roc_curve_each_model.png'))
    roccurveeachmodelurl = 'http://localhost:5000/graph/?name=roc_curve_each_model.png'
    return roccurveeachmodelurl
