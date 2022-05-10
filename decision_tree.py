import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

col_names = ['harga', 'partner', 'competitor', 'winlose']
# load dataset
dataset = pd.read_csv("training_data.csv", header=None, names=col_names, sep=';')

x = dataset.iloc[:, [1,2]].values
y = dataset.iloc[:, -1].values

rangeUnique = np.unique(x[:, 0])
winlossunique = np.unique(y)

# change value of x to index of unique value
for i in range(len(x)):
    for j in range(len(x[i])):
        # detect if the value is in the range of the unique values
        if x[i][j] in rangeUnique:
            x[i][j] = rangeUnique.tolist().index(x[i][j])

for i in range(len(y)):
    # detect if the value is in the range of the unique values
    if y[i] in winlossunique:
        y[i] = winlossunique.tolist().index(y[i])

x = x.astype(int)
y = y.astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0) # 70% training and 30% test

clf = DecisionTreeClassifier()

# # Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

# #Predict the response for test dataset
y_pred = clf.predict(x_test)

print("Win Probabilty:",metrics.accuracy_score(y_test, y_pred))