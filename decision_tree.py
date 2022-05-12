from cProfile import run
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.model_selection import cross_val_score
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

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

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(x_train,y_train)

#Predict the response for test dataset
# y_pred = clf.predict(x_test)
y_pred = clf.predict(x_test)

def run_model(new_x):

    
    # predict accuracy
    y_pred = clf.predict(new_x)
    winLossEvaluation = winlosscategory[y_pred[0]]

    # print(string)
    # print("Accuracy: {}%".format(accuracy))
    print("Win Loss Evaluation: ", winLossEvaluation)

    exit()

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

    arrayofInput = [harga_val, partner_val, competitor_val]

    new_x_test = [arrayofInput]
    run_model(new_x_test)

initialization()