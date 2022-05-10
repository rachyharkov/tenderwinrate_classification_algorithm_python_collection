import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("training_data.csv", sep=";")

x = dataset.iloc[:, [1,2]].values
y = dataset.iloc[:, -1].values

rangeUnique = np.unique(x[:, 0])
winlossunique = np.unique(y)

print(winlossunique)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

print(x_test)
print(y_test)

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

for i in range(len(y_train)):
        y_train[i] = winlossunique.tolist().index(y_train[i])

for i in range(len(y_test)):
        y_test[i] = winlossunique.tolist().index(y_test[i])

print(x_test)
print(y_test)

mdl = RandomForestRegressor()
mdl.fit(x_train, y_train)
y_predict = mdl.predict(x_test)
score = mdl.score(x_test, y_test)
print("Win Probabilty:", score)