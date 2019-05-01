import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
iris = load_iris()

# 75% 255 SPLIT: 75 TRAIN 25 TEST
#Returns 4 objects
#X_train - training set of data 75
#X_test - test set of data 25
#y_train - target content of training data
#y_test - target content of obsuring (testing)
#random_state = 0 - get same compnents (Use 0 for testing) : Use other numbers to randomize tests
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=100)
# n_neighbors - number of neighbors we want to coinsider (i.e n_neighbors = 1 is the closest neighbors 2 is the 2 cloest..)
knn = KNeighborsClassifier(n_neighbors=1)

#build the neighbor model on training data
knn.fit(X_train, y_train)

#New data - Data we  want to run our program on to see if it works! - example data !!! Needs to be wrapped in 2D array !!!
X_new = np.array([[5.0, 2.9, 1.0, 0.2]])

#.predict() - asks the knn for its prediction based on the training data : this variable will classify the species as [0], [1], or [2]
#prediction = knn.predict(X_new)

#this checks our programs correctness based on the 25% of the data we didn't train with Higher the better.
print(knn.score(X_test, y_test))
#0.9736842105263158
