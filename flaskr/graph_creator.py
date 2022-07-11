import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pydotplus
from sklearn.metrics import confusion_matrix
from sklearn import tree
import os

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
    labels = [0, 1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    # create heatmap
    _ = sns.heatmap(cnf_matrix, annot = True)
    ax.xaxis.set_label_position("top")
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Data Asli')
    plt.xlabel('Prediksi')

    fig.savefig(os.path.join(app.instance_path, 'graph_data', filename))
    confusionmatrixurl = 'http://localhost:5000/graph/?name=' + filename
    return confusionmatrixurl, cnf_matrix

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