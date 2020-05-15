__authors__ = ['1531206','1456135', '1533031']
__group__ = 'GrupDM12'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)


    def _init_train(self,train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """

        if (len(train_data.shape) == 3):
            self.train_data = train_data.reshape(
                (train_data.shape[0], train_data.shape[1]*train_data.shape[2])).astype('float64')
        else:
            # Calculate the RGB mean
            self.train_data = np.mean(train_data, axis=3).reshape(
                (train_data.shape[0], train_data.shape[1]*train_data.shape[2])).astype('float64')


    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data:   array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:  the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """

        if (len(test_data.shape) == 3):
            test_data = test_data.reshape(
                (test_data.shape[0], test_data.shape[1]*test_data.shape[2])).astype('float64')
        else:
            # Calculate the RGB mean
            test_data = np.mean(test_data, axis=3).reshape(
                (test_data.shape[0], test_data.shape[1]*test_data.shape[2])).astype('float64')

        distances = cdist(test_data, self.train_data).argsort(kind='mergesort')[:,:k]
        self.neighbors = self.labels[distances]


    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        
        # classes = []
        # for image in self.neighbors:
        #     _, indices, count = np.unique(image, return_index=True, return_counts=True)
        #     sorted_indices = np.sort(indices)
        #     count = count[indices.argsort()]
        #     classes.append(image[sorted_indices[np.argmax(count)]])

        classes, indices = np.unique(self.neighbors, return_inverse=True)
        indices_rs = indices.reshape(self.neighbors.shape)
        count = np.apply_along_axis(np.bincount, 1, indices_rs, None, np.max(indices) + 1)
        classes = classes[np.argmax(count, 1)].astype('<U8')
        return np.array(classes)


    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix ( N points in a D dimensional space)
        :param k:         :param k:  the number of neighbors to look at
        :return: the output from get_class (2 Nx1 vector, 1st the classm 2nd the  % of votes it got
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
