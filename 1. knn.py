# ## 와인 데이터 ##
#
# #데이터 처리
# from sklearn import datasets
# iris = datasets.load_wine()
# print(iris)
#
# from sklearn.model_selection import train_test_split
#
# x=iris.data
# y=iris.target
#
# print(x)
# print(y)
#
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=4)
#
# print(x_train.shape)
# print(x_test.shape)
#
# #데이터 학습
# from sklearn.neighbors import KNeighborsClassifier
#
# knn = KNeighborsClassifier(n_neighbors=40)
# knn.fit(x_train,y_train)
#
# #예측 및 평가
# y_pred = knn.predict(x_test)
# from sklearn import metrics
# scores = metrics.accuracy_score(y_test, y_pred)
# print(scores)


#################################################################################

## MNIST ##



# import matplotlib.pyplot as plt
# from sklearn import datasets, metrics
# from sklearn.model_selection import train_test_split
#
# digits = datasets.load_digits()
# plt.imshow(digits.images[0], cmap="gray", interpolation='nearest')
# plt.show(block=True)
#
# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))


# X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2)
#
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=3)
#
# knn.fit(X_train, y_train)
#
# y_pred = knn.predict(X_test)
# scores = metrics.accuracy_score(y_test, y_pred)
# print(scores)

import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
import seaborn as sn
import pandas as pd
import numpy
from sklearn.neighbors import KNeighborsClassifier


digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


knn = KNeighborsClassifier(n_neighbors=6)
 test_size=0.2)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.show()



