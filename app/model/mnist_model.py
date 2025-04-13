#this is starting

# load dataset 
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd

(x_train,y_train),(x_test,y_test)=mnist.load_data()

# show an example
# print(type(mnist.load_data()))
# print((pd.DataFrame(x_train[0])[0].unique()))
# print(type(x_train))

plt.imshow(x_train[0],cmap='gray')

plt.title(f'Label:{y_train[0]}')

plt.show()

# normalize the picture size

x_train=x_train/255.0

x_test=x_test/255.0

# reshape for input to a dense NN
x_train=x_train.reshape(-1,28*28)
x_test=x_test.reshape(-1,28*28)

# build the model

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


model=Sequential([
    Dense(128,activation='relu',input_shape=(784,)),
    Dense(64,activation='relu'),
    Dense(10,activation='softmax')# 10 output classes
])

# compile and train
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=5,validation_split=0.2)


# evaluate

test_loss,test_acc=model.evaluate(x_test,y_test)

print(f'test_acc: {test_acc}')

# predict

import numpy as np
pred=model.predict(x_test[0:1])
print(f'predicted :{np.argmax(pred)},actual: {y_test[0]}')

# save and load

model.save('mnist_model.h5')

# load model
from tensorflow.keras.models import load_model
loaded_model=load_model('mnist_model.h5')
