import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

print(x)
print(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))


# feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print(x_train)

logisticReg = LogisticRegression()
logisticReg.fit(x_train, y_train)
logisticReg.predict(x_test[0].reshape(1, -1))
logisticReg.predict(x_test[0:10])
predictions = logisticReg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logisticReg.score(x_test, y_test)))

def initialization():
    print("-------------- DECISION TREE PREDICTION - UNDER DEVELOPMENT --------------")
    print("Please, enter the following data based on range:")
    print("0 = Medium")
    print("1 = Moderate")
    print("2 = Optimis")
    print("-----------------------------------------------------")
    harga_val = input("Harga : ")
    partner_val = input("Partner : ")
    competitor_val = input("Competitor : ")

    arrayofinput = np.array([[harga_val, partner_val, competitor_val]])

    new_x_test = sc.transform(arrayofinput)
    print(new_x_test)
    # predict the result
    newprediction = logisticReg.predict(new_x_test)
    
    print("Win Loss Evaluation: ", winlosscategory[newprediction[0]])
    # probability of the result
    prob = logisticReg.predict_proba(new_x_test)[0][0]
    # probability of win
    probwin = logisticReg.predict_proba(new_x_test)[0][1]
    print("Probability of win: ", probwin * 100, "%")
    print("Probability of loss: ", (prob - probwin) * 100, "%")
    print("-----------------------------------------------------")
    exit()

initialization()