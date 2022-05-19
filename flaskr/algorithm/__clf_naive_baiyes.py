from cProfile import run
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
cols = ['harga', 'partner', 'competitor', 'winrate']

dataset = pd.read_csv("training_data.csv", sep=";",header=None, names=cols)


x = dataset.iloc[:, :3].values
y = dataset.iloc[:, -1].values

print(x)
print(y)

winlosscategory = np.unique(y)
# get the unique values of categorical data
rangeUnique = np.unique(x[:, 0])
print(rangeUnique)
# then change x based on the unique values
for i in range(len(x)):
    for j in range(len(x[i])):
        # detect if the value is in the range of the unique values
        if x[i][j] in rangeUnique:
            x[i][j] = rangeUnique.tolist().index(x[i][j])


# convert y to binary
y = np.where(y == "Win", 1, 0)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


# feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# print(x_train)

clf = GaussianNB()

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)