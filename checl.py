import numpy as np
import random
from random import shuffle
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.core import fully_connected,input_data,dropout
from tflearn.layers.estimator import regression
from tflearn.layers.conv import max_pool_2d,conv_2d
TRAIN_DIR=r'C:\Users\naman\Desktop\image_AI\train'
TEST_DIR=r'C:\Users\naman\Desktop\image_AI\test'
IMG_SIZE=50
LR=1e-3
MODEL_NAME='dogs_vs_cats.model'.format(LR,'2conv-basic-video')

def label(img):
	path_label=img.split('.')[-3]
	if path_label=='cat':
		return [1,0]
	elif path_label=='dog':
		return [0,1]
def set_train_data():
	training_data=[]
	for img in tqdm(os.listdir(TRAIN_DIR)):
		labels=label(img)
		path=os.path.join(TRAIN_DIR,img)
		img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
		training_data.append([np.array(img),np.array(labels)])
	shuffle(training_data)
	return training_data


def set_test_data():
	testing_data=[]
	for img in tqdm(os.listdir(TEST_DIR)):
		img_num=img.split[0]
		path=os.path.join(TEST_DIR,img)
		img=cv2.resize(cv2.imread(path,IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
		testing_data.append([np.array(img),np.array(img_num)])
	return testing_data

convnet=input_data(shape=[None,IMG_SIZE,IMG_SIZE,1],name='input')
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,2*32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,2*32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=fully_connected(convnet,1024,activation='relu')
convnet=dropout(convnet,0.8)
convnet=fully_connected(convnet,2,activation='softmax')
convnet=regression(convnet,learning_rate=LR,optimizer='adam',loss='categorical_crossentropy',name='targets')
model=tflearn.DNN(convnet,tensorboard_dir='log')
train_data=set_train_data()
train=train_data[:-500]
test=train_data[-500:]
x=np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y=np.array([i[1] for i in train])
test_x=np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y=np.array([i[1] for i in test])
model.fit({'input':x},{'targets':y},n_epoch=5,validation_set=({'input':test_x},{'targets':test_y}),show_metric=True,snapshot_step=1000)
#tensorboard --logdir=foo:C:\Users\naman\Desktop\image_AI\log

test_data=set_test_data()
for data,num in enumerate(test_data[:20]):
	img_data=data[0]
	img_num=data[1]
	orig=img_data
	y=fig.add_subplot(4,5,num+1)
	data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)
	model_out=model.predict([data])[0]
	if np.argmax(model_out)==1:str_label='Dog'
	else : str_label='Cat'
	y.imshow(orig,cmap='gray')
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
plt.show()

