#1-layer network for handwritten digits recognition

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

logPath = "./tb_logs/"

def variable_summaries(var):
   with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #reads data into the chosen directory

sess = tf.InteractiveSession()

with tf.name_scope('MNISTinput'):
    #placeholder for 28x28 grayscale image
    x = tf.placeholder(tf.float32, shape=[None, 784]) #None - we don't know how many items in this dimension

    #placeholder for probability of recognizing an image as a specific digit
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

with tf.name_scope('reshape'):
    # change data into 2D
    xImage = tf.reshape(x,[-1,28,28,1], name="x_image") #-1 appropriate data dimension so that it all fits
    tf.summary.image("inputImg",xImage,5)

#funtion for creating weight&bias variables
def weightVariable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1) # relu requires some non-zero values -> random numbers with 0.1 deviation, too big magnitude-> dropped
    return tf.Variable(initial, name=name)

def biasVariable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x,weights, name=None):
    return tf.nn.conv2d(x,weights,strides=[1, 1, 1, 1],padding = 'SAME', name=name) #strdes [0] and [3] must be 1

def maxPool2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


#defining layers

# 1st -> 5x5 filter for 32 features
with tf.name_scope('Conv1'):
    with tf.name_scope('weights'):
        weightsConv1 = weightVariable([5,5,1,32], name="weight") # 1 for number of input channels
        variable_summaries(weightsConv1)
    with tf.name_scope('biases'):
        biasesConv1 = biasVariable([32], name="bias")
        variable_summaries(biasesConv1)
    Conv1 = conv2d(xImage,weightsConv1) + biasesConv1
    tf.summary.histogram('conv1',Conv1)
    yConv1 = tf.nn.relu(Conv1, name="relu")
    tf.summary.histogram('yConv1', yConv1)
    yPool1 = maxPool2x2(yConv1, name="pool")

# 2nd -> 5x5 filter for 64 features
with tf.name_scope('Conv2'):
    with tf.name_scope('weights'):
        weightsConv2 = weightVariable([5,5,32,64], name="weight") # 1 for number of input channels
        variable_summaries(weightsConv2)
    with tf.name_scope('biases'):
        biasesConv2 = biasVariable([64], name="bias")
        variable_summaries(biasesConv2)
    
    Conv2 = conv2d(yPool1,weightsConv2) + biasesConv2
    tf.summary.histogram('Conv2',Conv2)
    yConv2 = tf.nn.relu(Conv2, name="relu")
    tf.summary.histogram('yConv2', yConv2)
    yPool2 = maxPool2x2(yConv2, name="pool")

# 3rd - fully connected
with tf.name_scope('FC1'):
    weightsFC1 = weightVariable([7*7*64,1024], name="weight") #shape-> [inputs,number of neurons]
    biasesFC1 = biasVariable([1024], name="bias")

    yPool2flat = tf.reshape(yPool2,[-1, 7*7*64])
    yFC1 = tf.nn.relu(tf.matmul(yPool2flat,weightsFC1)+biasesFC1, name="relu")

with tf.name_scope('dropout'):
    # dropout neurons for avoiding overfittig
    keepProb = tf.placeholder(tf.float32)
    yFC1drop = tf.nn.dropout(yFC1, keepProb)

with tf.name_scope('output'):
    # 4th - fully connected
    weightsFC2 = weightVariable([1024,10])
    biasesFC2 = biasVariable([10])

    yFC2 = tf.matmul(yFC1drop,weightsFC2)+biasesFC2

#cost function
with tf.name_scope("crossEntropy"):
    crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=yFC2))

with tf.name_scope("loss_optmizer"):
    trainStep = tf.train.AdamOptimizer(1e-4).minimize(crossEntropy)

with tf.name_scope("accuracy"):
    correctPrediction = tf.equal(tf.argmax(yFC2,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

tf.summary.scalar("crossEntropy", crossEntropy)
tf.summary.scalar("accuracy", accuracy)

summarizeAll = tf.summary.merge_all()

# Initialize all of the variables
sess.run(tf.global_variables_initializer())

#tensor board
tbWriter = tf.summary.FileWriter(logPath, sess.graph)

# Train the model
import time

#  define number of steps and how often we display progress
num_steps = 100
display_every = 10

# Start timer
start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    _,summary = sess.run([trainStep,summarizeAll], feed_dict={x: batch[0], y_: batch[1], keepProb: 0.5})

    # Periodic status display
    if i%display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keepProb: 1.0})
        end_time = time.time()
        print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy*100.0))
        tbWriter.add_summary(summary,i)

# Display summary 
#     Time to train
end_time = time.time()
print("Total training time for {0} batches: {1:.2f} seconds".format(i+1, end_time-start_time))

#     Accuracy on test data
print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keepProb: 1.0})*100.0))

sess.close()
