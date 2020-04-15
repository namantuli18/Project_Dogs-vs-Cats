import cv2
from tqdm import tqdm
import os
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,fully_connected,dropout
from tflearn.layers.estimator import regression
from random import shuffle
import numpy as np
DR=0.8
LR=1e-3
TRAIN_DIRECTORY=r"C:\Users\naman\Desktop\image_AI\train"
TEST_DIRECTORY=r"C:\Users\naman\Desktop\image_AI\test"
img_size=50
IMG_SIZE=50
def img_lablef(img):
	img_lable=img.split('.')[-3]
	if img_lable=='cat':
		return [1,0]
	elif img_lable=='dog':
		return [0,1]
def set_training_data():
	training_data=[]
	for img in tqdm(os.listdir(TRAIN_DIRECTORY)):
		label=img_lablef(img)
		path=os.path.join(TRAIN_DIRECTORY,img)
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(img_size,img_size))
		training_data.append([img,label])
	shuffle(training_data)
	return training_data
def set_testing_data():
	testing_data=[]
	for img in tqdm(os.listdir(TEST_DIRECTORY)):
		img_num=img.split('.')[0]
		path=os.path.join(TRAIN_DIRECTORY,img)
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(img_size,img_size))
		testing_data.append([img,img_num])
	return testing_data
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet,tensorboard_dir='log')
train_data=set_training_data()
train=train_data[:-500]
test=train_data[-500:]
x=np.array([i[0] for i in train]).reshape(-1,img_size,img_size,1)
y=np.array([i[1] for i in train])
test_x=x=np.array([i[0] for i in test]).reshape(-1,img_size,img_size,1)
test_y=np.array([i[1] for i in test])
model.fit({'input':x},{'targets':y},show_metric=True,snapshot_step=350,validation_set=({'input':test_x},{'targets':test_y}),n_epoch=5)




