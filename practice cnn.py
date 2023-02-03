from keras.datasets import mnist
from keras import models
from keras import layers
#
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#
# digit = train_images[0]
# digit.shape
#
#
# import matplotlib.pyplot as plt
#
# digit = train_images[0]
# plt.imshow(digit)
#
#
# network = models.Sequential() #모델의 레리어를 선형으로 연결하여 구성한다.
# network.add(layers.Conv2D(28,(3,3), activation = 'relu', input_shape=(28,28,3))) #모델의 첫 번째 레이어는 입력 형대에 대한 정보를 받는다.(input_image)크기 / 노드는 512개로 설정하고 활성화 함수는 reLu
# network.add(layers.Dense(10,activation = 'softmax'))# MNIST는 10개의 정답을 갖고 있기 때문에 output 노드 10개로 설정 / 활성화 함수는 확률  값을 갖게하는 softmax 사용
#
# network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy']) # 모델을 학습시키기 이전에 학습 방식에 대한 환경설정 / 보통 optimizer은 rmsprp나 adagrad 사용
#                                                                                                 # 손실함수는 모델이 최적화에 사용되는 목적함수 categorical_crossentropy or mse 사용
#                                                                                                 # metrics는 분류 문제이므로 acuuracy로 설정
# network.fit(train_images, train_labels, epochs=5, batch_size=130) #train_images 와 train_labels 사용하고 반복은 5번(전체 데이터 도는데) 데이터130개씩 학습 1300개이면 10으로 나누고 5번 반복


####################################################################################################################
import os, glob
import cv2
import numpy as np
from skimage.transform import pyramid_reduce
from tqdm import tqdm
import matplotlib.image

base_path = r'C:/Users/오용석/desktop/celeba-dataset'
img_base_path = os.path.join(base_path, 'img_align-celeba')
target_img_path = os.path.join(base_path, 'processed')

eval_list = np.loadtxt(os.path.join(base_path,'list_eval_partition.csv'), dtype=str, delimiter=',', skiprows=1)

x=2
