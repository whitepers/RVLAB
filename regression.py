# import numpy as np
# import matplotlib.pyplot as plt
# x = np.array([0.0, 1.0, 2.0])
# y = np.array([3.0, 3.5, 5.5])
#
# w=0  #가중치 0으로 설정
# b=0  #바이어스 0으로 설정
#
# Irate = 0.001 #학습율 0.01로 설정
# epochs = 100 # 반복 1000번
#
# n = float(len(x)) #x의 길이 확인 n=3
#
# #경사 하강법
# for i in range(epochs):  #1000번 반복
#     y_pred = w*x + b #선형 회귀 예측 값 , [0,0,0] 으로 시작
#     dw = (2/n) * sum(x*(y_pred-y))
#     db = (2/n) * sum(y_pred-y)
#     w = w-Irate*dw #가중치 수정
#     b = b-Irate*db #바이어스 수정
#
# print(w,b) #경사하강법을 통해 최적의 바이어스와 가중치를 찾아 출력
#
# y_pred = w*x + b #위에서 찾은 최적의 바이어스와 가중치 값과 x값을 통해 예측값 만듬
# plt.scatter(x,y) #그래프상에 출력
# plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color ='red') #예측 값은 선 그래프로 출력
# plt.show()

########################################################################

# import matplotlib.pyplot as plt
# from sklearn import linear_model
# reg = linear_model.LinearRegression() #선형회귀 모델 생성
#
# x = [[174],[152], [138], [128], [186]]
# y = [71, 55, 46, 38, 88]
#
# reg.fit(x,y) #위 x,y 데이터로 학습
# print(reg.predict([[165]])) # x가 165일 때 예측 값 프린트
#
# plt.scatter(x,y, color = 'black')
# y_pred = reg.predict(x) #reg.fit에서 경사하강법을 통해 최적의 바이어스와 가중치를 얻고 그 직선에 x값을 대입해서 예측값을 얻어냄
# plt.plot(x,y_pred, color = 'blue', linewidth=3)#계산된 w,b를 갖는 직선 그려짐
# plt.show()

########################################################################

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import linear_model
# from sklearn import datasets
#
#
# diabetes_x, diabetes_y = datasets.load_diabetes(return_X_y=True) #당뇨병 데이터 세트 x,y로 분리
# diabetes_x_new = diabetes_x[:, np.newaxis, 2] #하나의 특징만 선택해서 2차원 배열로 만든다. ex) age, bmi ...
#
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(diabetes_x_new, diabetes_y, test_size=0.1, random_state=0) #train set, test set 분리
#
# regr = linear_model.LinearRegression() #모델 생성
# regr.fit(x_train, y_train) # 선형회귀로 x_train 와 y_train 학습
#
# y_pred =regr.predict(x_test) # 학습으로 최적의 가중치와 바이어스를 얻고 x_test 값을 대입해서 y_pred 값을 얻음
# plt.plot(y_test, y_pred,'.') # 실제 데이터와 예측 데이터를 비교
#
# plt.xlim([-0.075, 0.100]) #x축이 - 값을 갖으므로 설정
# plt.scatter(x_test, y_test, color ='black')
# plt.plot(x_test, y_pred, color= 'blue', linewidth=3)
# plt.show()

########################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('C:/Users/오용석/Desktop/catdog/data.csv')
df.head()
x=np.array(df['32.50234527']) #데이터 찍어봤을때 한 칸 당겨짐..
y=np.array(df['31.70700585']) #데이터 찍어봤을때 한 칸 당겨짐..

def CostFunction1(x,y,a,b): #a = w , b = b
    cost=0 # cost =0으로 초기화
    for i in range(len(y)):
        cost+=(a*x[i]+b-y[i])**2
    return cost/2/len(y)    #for문을 통한 costfunction 계산

def CostFunction2(x,y,a,b):
    return sum((x*a+b-y)**2)/2/len(y)


#constfunction 1,2 는 같음
print(CostFunction1(x,y,1,10)) #초기 가중치 1 바이어스 10으로 설정
print(CostFunction2(x,y,1,10)) #초기 가중치 1 바이어스 10으로 설정


#경사 하강법을 위한 함수
def Grad_a(x, y, a, b): # a=w, b=b , 기울기가 음수
    return sum((a * x + b - y) * x) / len(y)
def Grad_b(X, y, a, b): # a=w, b=b , 기울기가 양수
    return sum(a * x + b) / len(y)


#훈련 df의 데이터 x,y와 반복 횟수와 학습률을 사용
def Train(X,y,Iteration,LearningRate):
    a,b=0,0
    for _ in range(Iteration):
        g_a=Grad_a(x,y,a,b)
        g_b=Grad_b(x,y,a,b)
        a-=LearningRate*g_a
        b-=LearningRate*g_b
    return [a,b]

a,b=Train(x,y,1000,0.0001)
plt.scatter(x,y)
plt.plot([min(x),max(x)],np.array([min(x),max(x)])*a+b)
plt.show()
# 학습율 변경
a,b=Train(x,y,1000,0.001)
plt.scatter(x,y)
plt.plot([min(x),max(x)],np.array([min(x),max(x)])*a+b)
plt.show()
# 학습율 변경
a,b=Train(x,y,1000,0.0000001)
plt.scatter(x,y)
plt.plot([min(x),max(x)],np.array([min(x),max(x)])*a+b)
plt.show()

########################################################################