# import pandas as pd
# import numpy as np
# df= pd.read_excel('C:/Users/오용석/Desktop/catdog/ex3d1.xlsx', 'X', header=None)
# df.head()
#
# import matplotlib.pyplot as plt
# plt.imshow(np.array(df.iloc[500, :]).reshape(20,20)) # df에 저장된 500번째 행과 모든 열 출력한 후 20x20 으로 바꿔준다. 행이 데이터의 특징이 되고  열이 데이터의 갯수가 된다.
# plt.show()
#
# plt.imshow(np.array(df.iloc[1750, :]).reshape(20,20)) # df에 저장된 1750번째 행과 모든 열 출력한 후 20x20 으로 바꿔준다.
# plt.show()
#
# print(len(df))
#
# df_y= pd.read_excel('C:/Users/오용석/Desktop/catdog/ex3d1.xlsx', 'y', header=None)
# df_y.head()
#
# y = df_y[0] # 10
# for i in range(len(y)):
#     if y[i] != 1:
#         y[i] = 0
#
# x_train = df.iloc[0:4000].T
# y_train = y.iloc[0:4000].T
# x_test = df.iloc[4000:].T
# y_test = y.iloc[4000:].T
#
# x_train = np.array(x_train)
# y_train = np.array(y_train)
# x_test = np.array(x_test)
# y_test = np.array(y_test)
#
#
# def sigmoid(z):
#     s = 1/(1 + np.exp(-z))
#     return s
# def initialize_with_zeros(dim):
#     w = np.zeros(shape=(dim,1))
#     b = 0
#     return w, b
# def propagate(w, b, X, Y):
#     #Find the number of training data
#     m = X.shape[1]
#     #Calculate the predicted output
#     A = sigmoid(np.dot(w.T, X) + b)
#     #Calculate the cost function
#     cost = -1/m * np.sum(Y*np.log(A) + (1-Y) * np.log(1-A))
#     #Calculate the gradients
#     dw = 1/m * np.dot(X, (A-Y).T)
#     db = 1/m * np.sum(A-Y)
#     grads = {"dw": dw, "db": db}
#     return grads, cost
# def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
#     costs = []
#     #propagate function will run for a number of iterations
#     for i in range(num_iterations):
#         grads, cost = propagate(w, b, X, Y)
#         dw = grads["dw"]
#         db = grads["db"]
#         #Updating w and b by deducting the dw
#         #and db times learning rate from the previous
#         #w and b
#         w = w - learning_rate * dw
#         b = b - learning_rate * db
#         #Record the cost function value for each 100 iterations
#         if i % 100 == 0:
#             costs.append(cost)
#         #The final updated parameters
#         params = {"w": w,
#               "b": b}
#         #The final updated gradients
#         grads = {"dw": dw,
#              "db": db}
#         return params, grads, costs
# def predict(w, b, X):
#     m = X.shape[1]
#     #Initializing an aray of zeros which has a size of the input
#     #These zeros will be replaced by the predicted output
#     Y_prediction = np.zeros((1, m))
#     w = w.reshape(X.shape[0], 1)
#     #Calculating the predicted output using the Formula 1
#     #This will return the values from 0 to 1
#     A = sigmoid(np.dot(w.T, X) + b)
#     #Iterating through A and predict an 1 if the value of A
#     #is greater than 0.5 and zero otherwise
#     for i in range(A.shape[1]):
#         Y_prediction[:, i] = (A[:, i] > 0.5) * 1
#     return Y_prediction
# def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5):
#     #Initializing the w and b as zeros
#
#     w, b = initialize_with_zeros(X_train.shape[0])
#     parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=False)
#     w = parameters["w"]
#     b = parameters["b"]
#     # Predicting the output for both test and training set
#     Y_prediction_test = predict(w, b, X_test)
#     Y_prediction_train = predict(w, b, X_train)
#     #Calculating the training and test set accuracy by comparing
#     #the predicted output and the original output
#     print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
#     print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
#     d = {"costs": costs,
#          "Y_prediction_test": Y_prediction_test,
#          "Y_prediction_train" : Y_prediction_train,
#          "w" : w,
#          "b" : b,
#          "learning_rate" : learning_rate,
#          "num_iterations": num_iterations}
#     return d
#
# d = model(x_train, y_train, x_test, y_test, num_iterations = 2000, learning_rate = 0.01)
# print(d)
##################################################################################################################
# import math
# import numpy as np
# import pandas as pd
# from pandas import DataFrame
# from sklearn import preprocessing
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from numpy import loadtxt, where
# from pylab import scatter, show, legend, xlabel, ylabel
#
# #2class csv파일을 이용한 logistic regression 문제
# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
# df=pd.read_csv('C:/Users/오용석/Desktop/catdog/data.csv')
#
# # df.columns = ["grade1","grade2","label"]
#
#
# X = df[["grade1","grade2"]]
# X = np.array(X) #특징만 x에 저장하고 라벨은 저장 안함
# X = min_max_scaler.fit_transform(X) #정규화 과정을 거침
# Y = df["label"]
# Y = np.array(Y)
#
# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33)
#
# clf = LogisticRegression()
# clf.fit(X_train,Y_train)
# print ('score Scikit learn: ', clf.score(X_test,Y_test))
#
# label1 = where(Y == 1)
# label0 = where(Y == 0)
# scatter(X[label1, 0], X[label1, 1], marker='o', c='b')
# scatter(X[label0, 0], X[label0, 1], marker='x', c='r')
# xlabel('Exam 1 score')
# ylabel('Exam 2 score')
# show()
###################################################################################
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
# y_data = [[0], [0], [0], [1], [1], [1]]
#
# x_train = torch.FloatTensor(x_data)
# y_train = torch.FloatTensor(y_data)
#
# print(x_train.shape)
# print(y_train.shape)
#
# W = torch.zeros((2, 1), requires_grad=True) # x_train 6x2 인데 y_train이 6X1이므로 가중치는 2X1이 되어야함
# b = torch.zeros(1, requires_grad=True)
#
# #torch.sigmoid와 같음
# hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W) + b)))
#
# losses = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis))
#
#
# # 모델 초기화
# W = torch.zeros((2, 1), requires_grad=True)
# b = torch.zeros(1, requires_grad=True)
# # optimizer 설정
# optimizer = optim.SGD([W, b], lr=1)
#
# nb_epochs = 1000
# for epoch in range(nb_epochs + 1):
#     # Cost 계산
#     hypothesis = torch.sigmoid(x_train.matmul(W) + b)
#     cost = -(y_train * torch.log(hypothesis) +
#              (1 - y_train) * torch.log(1 - hypothesis)).mean()
#     # cost로 H(x) 개선
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#     # 100번마다 로그 출력
#     if epoch % 100 == 0:
#         print('Epoch {:4d}/{} Cost: {:.6f}'.format(
#             epoch, nb_epochs, cost.item()
#         ))
######################################################################################
import pandas as pd
from sklearn.metrics import accuracy_score
passengers = pd.read_csv('C:/Users/오용석/Desktop/catdog/tested.csv')

print(passengers.shape)
print(passengers.head())

passengers['Sex'] = passengers['Sex'].map({'female':1,'male':0})
passengers['Age'].fillna(value=passengers['Age'].mean(), inplace=True)
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']
from sklearn.model_selection import train_test_splitㅋ
train_features, test_features, train_labels, test_labels = train_test_split(features, survival)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(train_features, train_labels)
predict = model.predict(test_features)
predict = predict.reshape(-1,1)
accuracy_score(predict, test_labels)

x=2

