import tensorflow as tf
from keras import backend as k
#For creating the Sequential model(Neural_Network)
from keras.models import Sequential
#Core layers of keras
from keras.layers import Dense, Dropout, Activation, Flatten

#For 2D Convolution
from keras.layers import Conv2D, MaxPooling2D

#importing the dataset
from keras.datasets import mnist

#We use this to create sparse categorical matrix
from keras.utils import np_utils


#Loading the mnist image data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)

#Image shape is 28 X 28.

img_rows, img_cols = 28, 28

import matplotlib.pyplot as plt

#How the actual image looks
plt.imshow(X_train[0],'gray')
plt.show()

# Channels_first => 'channels', 'rows', 'columns'
# Channels_last  => 'rows', 'columns', 'channels'

#This prevents dimension mismatch error
if k.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


print(X_train.shape)
# Float conversion
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# To fit to the RGB values
X_train/= 255
X_test/= 255

print(y_train.shape)

#Converting target data into categorical data of classes (0-9)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

print(y_train.shape)
print(y_train[0])

#Creating the network
model = Sequential()
#Convolution layer1
model.add(Conv2D(32, (3, 3), strides = (1,1), activation='relu', input_shape = input_shape))
#Convolution layer2
model.add(Conv2D(32, (3, 3), strides = (1,1), activation='relu'))

#Max pooling is applied, you can use average pooling also
model.add(MaxPooling2D(pool_size = (2,2)))

#Dropout is applied to prevent overfitting and create an ensemble network
model.add(Dropout(0.2))

#Flattens the output after the Maxpooling is applied. This transformation is needed to give it as input to the Neural network
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))


#Configure the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#Training the model
model.fit(X_train, y_train, batch_size = 128, epochs = 1, verbose = 2)

#Evaluating the model on test_data
score = model.evaluate(X_test, y_test, verbose = 0)
print('Evaluation score is : ')
print(score)
