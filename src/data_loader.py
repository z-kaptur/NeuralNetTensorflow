import scipy.io as sio
import tensorflow as tf
import numpy as np
import scipy 
from scipy import ndimage

class DataSet:


    def __init__ (self, dim_x, dim_y, dataset, train_path, test_path, train_labels=None, test_labels=None):
        self._dataset = dataset
        self._dim_x = dim_x
        self._dim_y = dim_y

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
        self._train_data = np.add(np.add(self._train_data[:,:,0]*0.299, self._train_data[:,:,1]*0.587),self._train_data[:,:,2]*0.114)
        self._test_data = np.add(np.add(self._test_data[:,:,0]*0.299, self._test_data[:,:,1]*0.587), self._test_data[:,:,2]*0.114)


    def prepare_shape(self, num_categories):
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
        return self._test_data[0:10], self._test_labels[0:10]
