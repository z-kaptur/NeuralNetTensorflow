"""Module for loading and preprocessing datasets"""

import scipy.io as sio
import tensorflow as tf
import numpy as np
import scipy 
from scipy import ndimage

class DataSet:
    
    def __init__ (self, height, width, dataset, train_path, test_path, train_labels=None, test_labels=None):
        """Construct a DataSet.
        
        Args:
            height: Images height.
            width: Images width.
            dataset: Name of the chosen dataset (mnist/svhn).
            train_path: Path to training data.
            test_path: Path to testing data.
            train_labels: Path to training data labels (only svhn dataset)
            test_labels: Path to testing data labels (only svhn dataset)
        """

        self._dataset = dataset
        self._height = height
        self._width = width

        if (dataset=="svhn"):
            train = sio.loadmat(train_path)
            train_data = train['X']
            train_data = train_data.astype(np.float32)
            train_data = np.multiply(train_data,1.0/255.0)
            train_data = np.add(train_data,-0.5)
            self._train_data = train_data
            self._train_labels = train['y'] 

            test = sio.loadmat(test_path)
            test_data = test['X']
            test_data = test_data.astype(np.float32)
            test_data = np.multiply(test_data,1.0/255.0)
            test_data = np.add(test_data,-0.5)
            self._test_data = test_data
            self._test_labels = test['y']

        elif(dataset=="mnist"):
            train_data = sio.loadmat(train_path)['data']
           # print(np.amax(train_data))
            train_data = train_data.astype(np.float32)
            train_data = np.multiply(train_data,1.0/255.0)
            self._train_data = train_data
            self._train_labels = sio.loadmat(train_labels)['data']

            test_data = sio.loadmat(test_path)['data']
            test_data = test_data.astype(np.float32)
            test_data = np.multiply(test_data,1.0/255.0)            
            self._test_data = test_data
            self._test_labels = sio.loadmat(test_labels)['data']

        self._epoch = 0
        self._current_index = 0
        self._new_epoch = False

    def cvt_grayscale(self):
        """Convert images data to grayscale."""

        self._train_data = np.add(np.add(self._train_data[:,:,0]*0.299, self._train_data[:,:,1]*0.587),self._train_data[:,:,2]*0.114)
        self._test_data = np.add(np.add(self._test_data[:,:,0]*0.299, self._test_data[:,:,1]*0.587), self._test_data[:,:,2]*0.114)

    def prepare_shape(self, num_categories):
        """Format data shape for processing in a neural network
        
        Args:
            num_categoies: Number of output categories for data classification.
        """

        with tf.Session() as sess:
            if (self._dataset == "svhn"):
                temp_train = self._train_data.reshape(self._train_data.shape[0]*self._train_data.shape[1],3,-1)
                self._train_data = temp_train.transpose((2,0,1))
                self._train_labels = tf.one_hot(self._train_labels,num_categories).eval().reshape(-1,10)
                temp_test = self._test_data.reshape(self._test_data.shape[0]*self._test_data.shape[1],3,-1)
                self._test_data = temp_test.transpose((2,0,1))
                self._test_labels = tf.one_hot(self._test_labels,num_categories).eval().reshape(-1,10)
                self.cvt_grayscale()
            elif (self._dataset == "mnist"):
                self._train_labels = tf.one_hot(self._train_labels,num_categories).eval().reshape(-1,10)
                self._test_labels = tf.one_hot(self._test_labels,num_categories).eval().reshape(-1,10)

    def load_batch(self, batch_num):
        """Load a batch of data from the dataset.

        Args:
            batch_num: Batch size.
        """

        num_all_samples = self._train_labels.shape[0]
        batch_size = batch_num
        if(self._new_epoch):
            self._new_epoch = False
            self._epoch = self._epoch + 1
            i = np.arange(int(num_all_samples))
            np.random.shuffle(i)
            self._train_data = self._train_data[i]
            self._train_labels = self._train_labels[i]
        elif(self._current_index + batch_num >= num_all_samples - 1):
            batch_size = num_all_samples - self._current_index
            self._new_epoch = True
        new_index = self._current_index + batch_size
        data = self._train_data[self._current_index:new_index]
        labels = self._train_labels[self._current_index:new_index]
        self._current_index = new_index % int(num_all_samples)
        return data, labels

    def load_test_data(self):
        """Load all test data"""

        return self._test_data, self._test_labels
