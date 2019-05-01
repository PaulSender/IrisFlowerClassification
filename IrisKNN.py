#Import data
from sklearn import datasets
iris = datasets.load_iris()

#features
print(iris.data)
#labels
print(iris.target)

#each entry has 4 attributes

#Splitting data into training and testing sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, = train_test_split(iris.data, iris.target, test_size = 0.33)

#0.33 means whole set has been divided in to two sets where test set is 33% of the original set and train set in 66% of the original test

#import knn classifier

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)

#----- trained classifer using training data -----
#knn is a lazy learner, it doesnt get trained, it takes all data with it when testing eaxh atteribute

#import accuracy metrics

from sklearn.metrics import accuracy_score
print("Accuracy is: ")
print(accuracy_score(y_test, clf.predict(x_test)))


# get accuracys from 94 to 98

# use k = 3, plot graph with k and accuracy

import matplotlib.pyplot as plt

# Iterate classifer and init it different k values and find accuracy

#accuracy values is 2D array, where each entry is [K, accuracy]
accuracy_values = []

# range() is xrange() in python 3
for x in range(1,x_train.shape[0]):
    clf = KNeighborsClassifier(n_neighbors=x).fit(x_train,y_train)
    accuracy = accuracy_score(y_test, clf.predict(x_test))
    accuracy_values.append([x,accuracy])
    pass

#convert normal python array to numpy array for efficient operations

import numpy as np
accuracy_values = np.array(accuracy_values)

plt.scatter(accuracy_values[:,0], accuracy_values[:,1])
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()

# Accuracy drops when k is more than 60
# K value selections depends on data distribution in this case, we have good accuracy when k lies between 40 and 60
