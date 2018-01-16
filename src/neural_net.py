import tensorflow as tf
import data_loader as dl
import nn_structures as nns
import shutil
import os
import scipy.io as sio
import argparse
import time

FLAGS = None

def main():
    log_path = "./tb_logs"
    if(os.path.isdir(log_path)): 
        shutil.rmtree(log_path)

    if FLAGS.dataset == "mnist":
        data = dl.DataSet(28,28,"mnist","../datasets/mnist/train_images.mat",
            "../datasets/mnist/test_images.mat",
            "../datasets/mnist/train_labels.mat",
            "../datasets/mnist/test_labels.mat")
        data.prepare_shape(FLAGS.num_categories)

    elif FLAGS.dataset == "svhn":
        data = dl.DataSet(32,32,"svhn", "../datasets/svhn_dataset/train_32x32.mat", 
            "../datasets/svhn_dataset/test_32x32.mat")
        data.prepare_shape(FLAGS.num_categories)

    data_height = data._height
    data_width = data._width

    num_dimensions = 1

    x = tf.placeholder(tf.float32,[None, data_height*data_width])
    y_ = tf.placeholder(tf.float32,[None, FLAGS.num_categories])

    output, keep_prob = nns.nn_4layers_ccff(x,num_dimensions,data_height,data_width,FLAGS.num_categories) 

    sess = tf.InteractiveSession()

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))

    with tf.name_scope("loss_optmizer"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("cross_entropy", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)

    summarize_all = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    tbWriter = tf.summary.FileWriter(log_path, sess.graph)

    start_time = time.time()
    end_time = time.time()

    for i in range(FLAGS.num_steps):
        batch_data, batch_labels = data.load_batch(FLAGS.batch_size)
        _,summary = sess.run([train_step,summarize_all], feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.5})

        if i%FLAGS.display_step == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch_data, y_: batch_labels, keep_prob: 1.0})
            end_time = time.time()
            print("step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time-start_time, train_accuracy*100.0))
            tbWriter.add_summary(summary,i)

    end_time = time.time()
    print("Total training time for {0} batches: {1:.2f} seconds".format(i+1, end_time-start_time))

    test_data, test_labels = data.load_test_data()
    print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
        x: test_data, y_: test_labels, keep_prob: 1.0})*100.0))

    saver = tf.train.Saver()
    saver.save(sess, '../trained_models/'+FLAGS.model_path)
    saver.export_meta_graph('../trained_models/'+FLAGS.model_path+'.meta')

    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--dataset',
      action='store',
      dest='dataset',
      default='mnist',
      help='Dataset: mnist or svhn.'
    )
    parser.add_argument(
      '--cat',
      action='store',
      dest='num_categories',
      type=int,
      default=1,
      help='Number of categories.'
    )
    parser.add_argument(
      '--batch',
      action='store',
      dest='batch_size',
      type=int,
      default=50,
      help='Batch size.'
    )
    parser.add_argument(
      '--steps',
      action='store',
      dest='num_steps',
      type=int,
      default=100,
      help='Number of steps.'
    )
    parser.add_argument(
      '--disp',
      action='store',
      dest='display_step',
      type=int,
      default=10,
      help='Display every ... steps.'
    )
    parser.add_argument(
      '--model',
      action='store',
      dest='model_path',
      type=str,
      default="model",
      help='Path for saving the model.'
    )
    FLAGS = parser.parse_args()
    main()