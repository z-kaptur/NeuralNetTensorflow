import tensorflow as tf

def conv_fun(yPool1):
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

        return yPool2
        

def weightVariable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1) # relu requires some non-zero values -> random numbers with 0.1 deviation, too big magnitude-> dropped
    return tf.Variable(initial, name=name)

def biasVariable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


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



def conv2d(x,weights, name=None):
    return tf.nn.conv2d(x,weights,strides=[1, 1, 1, 1],padding = 'SAME', name=name) #strdes [0] and [3] must be 1

def maxPool2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
