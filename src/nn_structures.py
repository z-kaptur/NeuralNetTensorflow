"""Functions for neural network structures"""

import tensorflow as tf

def variable_summaries(var):
    """Prepare variable summary for tensorboard"""

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
    """Generate random weights variable with a given shape"""

    init = tf.truncated_normal(shape, stddev=0.1) 
    return tf.Variable(init, name=name)


def generate_biases_variable(shape, name=None):
    """Generate biases variable with a given shape"""

    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init, name=name)


def conv_2d(x,weights, name=None):
    """Convolution of input with the filter definied by weights"""

    return tf.nn.conv2d(x,weights,strides=[1, 1, 1, 1],padding = 'SAME', name=name) #strdes [0] and [3] must be 1


def max_pool_2x2(x, name=None):
    """Max pooling, reduce dimensionality 2 times"""

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def convolutional_pooling_layer(input, filter_size, in_features_num, out_features_num, name=None):
    """A convolutional layer with max pooling.

    Args:
        input: Input tensor in 2D format.
        filter_size: Size of the filter applied in convolution phase.
        in_features_num: Number of features at the imput of the layer (1 if first layer).
        out_features_num: Number of features at the output of the layer.
        name: Name for analysis in tensorboard.

    Returns:
        Tensor with the output of the layer.
    """

    with tf.name_scope(name):
        with tf.name_scope('weights'):
            weights = generate_weights_variable (
                [filter_size[1],filter_size[0],in_features_num,out_features_num], name="weight") # 1 for number of input channels
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = generate_biases_variable([out_features_num], name="bias")
            variable_summaries(biases)

        conv = conv_2d(input,weights) + biases
        tf.summary.histogram('conv',conv)
        conv_activation= tf.nn.relu(conv, name="relu")
        tf.summary.histogram('conv+activation', conv_activation)
        pool = max_pool_2x2(conv_activation, name="pool")

        return pool
        

def fully_connected_layer(input, in_num, out_num, name=None, output=False):
    """A fully connected layer.

    Args:
        input: Input tensor in 2D or 1D format.
        in_num: Number of inputs to the layer.
        out_num: Number of neurons in the layer.
        name: Name for analysis in tensorboard.
        output: True if the last layer in the neural network. 

    Returns:
        Tensor with the output of the layer.
    """

    if(tf.rank(input)!=2):
        input_flat = tf.reshape(input,[-1, in_num])
    else:
        input_flat = input
  
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            weights = generate_weights_variable ([int(in_num), out_num], name="weight") # 1 for number of input channels
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = generate_biases_variable([out_num], name="bias")
            variable_summaries(biases)   
    
        f_c = tf.matmul(input_flat,weights)+biases
        if (not output): 
            f_c = tf.nn.relu(f_c)
        return f_c


def dropout(input, name=None):
    """Dropout of some neurons in a layer.

    Args:
        input: Input tensor in 1D format.
        name: Name for analysis in tensorboard.
       
    Returns:
        Tensor with the result of dropout.
        Placeholder for probability of keeping a neuron.
    """

    with tf.name_scope(name):
        keep_prob = tf.placeholder(tf.float32)
        drop = tf.nn.dropout(input, keep_prob)

    return drop, keep_prob


def nn_4layers_ccff (input, channels, height, width, categories_num):
    """4-layer neural network with two convolutional and two fully connected layers.

    Args:
        input: Input images in flat(1D) format.
        channels: Number of channels in an image.
        height: Height of input images.
        width: Width of input images.
        categories_num: Number of output categories.

    Returns:
        Tensor with the output of the network.
        Placeholder for probability od keeping a neuron in the dropout phase.
    """

    input_reshape = tf.reshape(input,[-1,height,width,channels], name="image")
    tf.summary.image("0",input_reshape,10)
    c1 = convolutional_pooling_layer(input_reshape,(5,5),1,32,"conv_1")
    c1_img = tf.reshape(c1[:,:,:,1],[-1,int(height/2),int(width/2),1],"c1_image")
    tf.summary.image("0",c1_img,10)
    c2 = convolutional_pooling_layer(c1,(5,5),32,64,"conv_2")
    f3 = fully_connected_layer(c2,int(height/4*width/4*64),1024,"fully_con_3")
    d3, keep_prob = dropout (f3, "dropout_3")
    output = fully_connected_layer(d3,1024,categories_num,"output", True)
    return output, keep_prob