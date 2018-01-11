#1-layer network for handwritten digits recognition

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as keepProb

logPath = "./tb_logs/"

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #reads data into the chosen directory

sess = tf.InteractiveSession()

imageRows = 28
imageCols = 28

print (mnist.train.images.shape)

trainImages = mnist.train.images.reshape(mnist.train.images.shape[0], imageRows, imageCols, 1)
testImages = mnist.test.images.reshape(mnist.test.images.shape[0], imageRows, imageCols, 1)

numFilters = 32
maxPoolSize = (2,2)
convKernelSize = (3,3)
imagShape = (28,28,1)
numClasses = 10
dropProb = 0.5

model = Sequential()

model.add(Convolution2D(numFilters, convKernelSize[0],convKernelSize[1], border_mode="valid", input_shape=imagShape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=maxPoolSize))

model.add(Convolution2D(numFilters, convKernelSize[0],convKernelSize[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=maxPoolSize))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(dropProb))

model.add(Dense(numClasses))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy",
            optimizer='adam',
            metrics=['accuracy'])

batchSize = 128
numEpoch = 2

model.fit(trainImages, mnist.train.labels, batch_size=batchSize, nb_epoch=numEpoch, verbose=1, validation_data=(testImages, mnist.test.labels))