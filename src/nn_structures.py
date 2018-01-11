import tensorflow as tf

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


def generate_weights_variable(shape, name=None):
    init = tf.truncated_normal(shape, stddev=0.1) 
    return tf.Variable(init, name=name)


def generate_biases_variable(shape, name=None):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init, name=name)

def conv_2d(x,weights, name=None):
    return tf.nn.conv2d(x,weights,strides=[1, 1, 1, 1],padding = 'SAME', name=name) #strdes [0] and [3] must be 1

def max_pool_2x2(x, name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def convolutional_pooling_layer(input, filter_size, in_features_num, out_features_num, name=None):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            weights = generate_weights_variable (
                [filter_size[1],filter_size[0],in_features_num,out_features_num], name="weight") # 1 for number of input channels
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = generate_biases_variable([out_features_num], name="bias")
            variable_summaries(biases)
        
        conv = conv_2d(input,weights) + biases
        tf.summary.histogram('conv',Conv2)
        conv_activation= tf.nn.relu(Conv2, name="relu")
        tf.summary.histogram('conv+activation', conv_activation)
        pool = max_pool_2x2(conv_activation, name="pool")

        return pool
        

def fully_connected_layer(input, in_num, out_num, name=None):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            weights = generate_weights_variable ([in_num, out_num], name="weight") # 1 for number of input channels
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = generate_biases_variable([out_num], name="bias")
            variable_summaries(biases)   
    
        f_c = tf.nn.relu(tf.matmul(input,weights)+biases, name="relu")
        
        return f_c


def dropout(input, name=None):
    with tf.name(name):
        keep_prob = tf.placeholder(tf.float32)
        drop = tf.nn.dropout(input, keep_prob)

    return drop, keep_prob

def nn_4layers_ccff (input, channels, dim_x, dim_y, labels, categories_num):
    input_reshape = tf.reshape(x,[-1,dim_x,dim_y,channels], name="image")
    c1 = convolutional_pooling_layer(input_reshape,(5,5),1,32,"conv_1")
    c2 = convolutional_pooling_layer(c2,(5,5),32,64,"conv_2")
    f3 = fully_connected_layer(c2,dim_x/4*dim_y/4*64,1024,"fully_con_3")
    d3, keep_prob = dropout (f3, "dropout_3")
    output = fully_connected_layer(d3,1024,categories_num)
    return output