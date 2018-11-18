#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 21:08:53 2018

@author: yashkumararora
"""

import numpy as np
import keras
from keras.layers import Input,Dense
from keras.models import Model
from keras.datasets import mnist

(x_train, y_tain),(x_test,y_test) = mnist.load_data()
one_hot_y = keras.utils.to_categorical(y_train)
x = x_train.reshape((60000,784))
inp = Input(shape=(784,))
hid1 = Dense(100,activation='sigmoid')(inp)
out = Dense(10,activation='sigmoid')(hid1)
model = Model(inputs=inp,outputs=out)
model.compile(optimizer='SGD',loss='MSE',metrics=['accuracy'])
model.fit(x,one_hot_y,epochs=5)
x_t = x_test.reshape((10000,784))
pred = model.predict(x_t)
c = 0

for i in range(0,len(y_test)):
    if(np.argmax(pred[i])==y_test[i]):
        c+=1

print (c)
print ('Total accuracy ',(float(c)*100.00)/len(y_test))