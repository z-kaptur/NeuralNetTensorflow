import tensorflow as tf
import numpy as np #scientific computing
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tflearn

#generating random house sizes
numHouse = 160
np.random.seed(23)
houseSize = np.random.randint(low=1000, high=3500, size=numHouse)

#generating house prices 
np.random.seed(23)
housePrice = houseSize * 100.0 + np.random.randint(low=20000, high=70000, size=numHouse)

#ploting data
plt.plot(houseSize, housePrice, 'bx')
plt.ylabel('Price')
plt.xlabel('Size')
plt.show()

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

