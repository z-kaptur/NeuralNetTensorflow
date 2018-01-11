import tensorflow as tf
import numpy as np #scientific computing
import math

from keras.models import Sequential
from keras.layers.core import Dense, Activation

#generating random house sizes
numHouse = 160
np.random.seed(23)
houseSize = np.random.randint(low=1000, high=3500, size=numHouse)

#generating house prices 
np.random.seed(23)
housePrice = houseSize * 100.0 + np.random.randint(low=20000, high=70000, size=numHouse)


def normalize(array):
    return (array - array.mean()) / array.std()

numTrainSamples = math.floor(numHouse * 0.7)

#training data
trainHouseSize = np.asarray(houseSize[:numTrainSamples])
trainHousePrice = np.asanyarray(housePrice[:numTrainSamples:]) # check the func

trainHouseSizeNorm = normalize(trainHouseSize)
trainHousePriceNorm = normalize(trainHousePrice)

#test data
testHouseSize = np.asarray(houseSize[numTrainSamples:])
testHousePrice = np.asanyarray(housePrice[numTrainSamples:])

testHouseSizeNorm = normalize(testHouseSize)
testHousePriceNorm = normalize(testHousePrice)

# defining keras model
model = Sequential()
model.add(Dense(1,input_shape=(1,),init="uniform", activation="linear"))
model.compile(loss='mean_squared_error',optimizer='sgd')

model.fit(trainHouseSizeNorm, trainHousePriceNorm, nb_epoch=300)

score = model.evaluate(testHouseSizeNorm,testHousePriceNorm)
print("\nloss on test: {0}".format(score))