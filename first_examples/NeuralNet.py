#1-layer network for handwritten digits recognition

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #reads data into the chosen directory

#placeholder for 28x28 grayscale image
x = tf.placeholder(tf.float32, shape=[None, 784]) #None - we don't know how many items in this dimension

#placeholder for probability of recognizing an image as a specific digit
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#variables that we're going to train
weights = tf.Variable(tf.zeros([784,10]))
biases = tf.Variable(tf.zeros([10]))

#defining model
y = tf.nn.softmax(tf.matmul(x, weights)+biases)

#cost function
crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

trainStep = tf.train.GradientDescentOptimizer(0.5).minimize(crossEntropy)

#initializint the variables 
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for iteration in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(trainStep,feed_dict={x: batch_xs, y_: batch_ys})


correctPrediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
testAccuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels})

print("test ACCURACY:{0}%".format(testAccuracy*100))

sess.close()