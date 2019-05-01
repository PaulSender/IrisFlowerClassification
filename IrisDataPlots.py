# IRIS classification

# Objective : classify iris flower based on dataset

#import datasets
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris = load_iris()
#print(iris.keys())
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
#print(iris)
#iris.data = actual data i.e lengths
#iris.target = array that holds the species labeled by 0,1, or 2
#print(iris.target_names)
#['setosa' 'versicolor' 'virginica']

#.T gives us a list of Lists that correspond to the featues of the data sets i.e - sepil L/W and petal L/W
features = iris.data.T

sepal_length = features[0]
sepal_width = features[1]
petal_length = features[2]
petal_width = features[3]

sepal_length_label = iris.feature_names[0]
sepal_width_label = iris.feature_names[1]
petal_length_label = iris.feature_names[2]
petal_width_label = iris.feature_names[3]

plt.scatter(sepal_length, sepal_width, c = iris.target)
plt.xlabel(sepal_length_label)
plt.ylabel(sepal_width_label)
plt.show()
