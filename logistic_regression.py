import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("training_data.csv", sep=";")

x = dataset.iloc[:, [0,2]].values
y = dataset.iloc[:, -1].values

rangeUnique = np.unique(x[:, 0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

for i in range(len(x_train)):
    for j in range(len(x_train[i])):
        # detect if the value is in the range of the unique values
        if x_train[i][j] in rangeUnique:
            x_train[i][j] = rangeUnique.tolist().index(x_train[i][j])

for i in range(len(x_test)):
    for j in range(len(x_test[i])):
        # detect if the value is in the range of the unique values
        if x_test[i][j] in rangeUnique:
            x_test[i][j] = rangeUnique.tolist().index(x_test[i][j])

# print(x_train)

logisticReg = LogisticRegression()

logisticReg.fit(x_train, y_train)
logisticReg.predict(x_test[0].reshape(1, -1))
logisticReg.predict(x_test[0:10])
predictions = logisticReg.predict(x_test)
print('Win Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticReg.score(x_test, y_test)))