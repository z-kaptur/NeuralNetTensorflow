import tensorflow as tf
import numpy as np #scientific computing
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

#setting up tensorflow placeholders for in/outs
tfHouseSize = tf.placeholder("float", name="house_size")
tfHousePrice = tf.placeholder("float", name="house_price")

#defining the variables for aproximating the curve
tfSizeFactor = tf.Variable(np.random.randn(), name="size_factor")
tfPriceOffset = tf.Variable(np.random.randn(), name="price_offset")

#prediction function 
tfPricePred = tf.add(tf.multiply(tfSizeFactor, tfHouseSize),tfPriceOffset)

#cost funcyion 
tfCost = tf.reduce_sum(tf.pow(tfPricePred-tfHousePrice,2))/(2*numTrainSamples)

learningRate = 0.1

optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(tfCost)

#initializint the variables 
init = tf.global_variables_initializer()

# launch the graph in the session
with tf.Session() as sess:
    sess.run(init)

    displayEvery = 2
    numTrainingIter = 50

    for iteration in range(numTrainingIter):

        for (x,y) in zip(trainHouseSizeNorm,trainHousePriceNorm):
            sess.run(optimizer,feed_dict={tfHouseSize: x, tfHousePrice: y})

        if (iteration+1) % displayEvery == 0:
            c = sess.run(tfCost, feed_dict = {tfHouseSize: trainHousePriceNorm, tfHousePrice: trainHousePriceNorm})
            print("iteration #:", '%04d' % (iteration+1), "cost=", "{:.9f}".format(c),"size_factor=", sess.run(tfSizeFactor),\
            "price_offset=", sess.run(tfPriceOffset))

    print("Optimiaion finished")
    trainingCost = sess.run(tfCost,feed_dict = {tfHouseSize:trainHouseSizeNorm, tfHousePrice:trainHousePriceNorm})
    print("Trained cost=",trainingCost,"size_factor=",sess.run(tfSizeFactor), "price offset=", sess.run(tfPriceOffset))
