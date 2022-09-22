import numpy as np


def preprocessing_data(X,y):

    for i in range(len(X)):
        for j in range(len(X[i])):

            X[i][j] = X[i][j].strip().replace(" ", "")
            X[i][j] = X[i][j].lower()

    # get the unique values of win result
    winlosscategory = np.unique(y)

    # get the unique values of categorical data
    rangeUnique = np.unique(X)

    newRangeUnique = []
    # reposition rangeUnique, move to index 0 if value is moderate, index 1 if values is medium, index 2 if value is optimis
    # as said in BAB 1 > Batasan Masalah > Poin 2
    for i in range(len(rangeUnique)):

        if rangeUnique[i] == "low":
            newRangeUnique.insert(0, rangeUnique[i])
        if rangeUnique[i] == "moderate":
            newRangeUnique.insert(1, rangeUnique[i])
        if rangeUnique[i] == "optimis":
            newRangeUnique.insert(2, rangeUnique[i])

    # then convert x based on the unique values
    for i in range(len(X)):
        for j in range(len(X[i])):
            # detect if the value is in the range of the unique values
            if X[i][j] in newRangeUnique:
                X[i][j] = newRangeUnique.index(X[i][j])


    # convert y to binary
    y = np.where(y == "Win", 1, 0)

    return X,y,winlosscategory