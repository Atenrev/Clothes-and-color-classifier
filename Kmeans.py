__authors__ = ['1531206','1456135', '1533031']
__group__ = 'GrupDM12'

import numpy as np
import pandas
import utils
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictºionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options


    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        if len(X.shape) == 2:
            self.X = X
        else:
            self.X = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))

        if self.X.dtype != np.float64:
            self.X = self.X.astype('float64')


    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options == None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  #within class distance.

        # If your methods need any other prameter you can add it to the options dictionary
        self.options = options


    def _init_centroids(self):
        """
        Initialization of centroids
        """

        if self.options['km_init'].lower() == 'first':
            _, idx = np.unique(self.X, return_index=True, axis=0)
            self.centroids = self.X[np.sort(idx)]
            self.centroids = self.centroids[:self.K]
        elif self.options['km_init'].lower() == 'random':
            self.centroids = np.random.rand(self.K, self.X.shape[1]) * 255
        elif self.options['km_init'].lower() == 'sharding':
            attributes_sum = np.sum(self.X, 1)
            ordering_indexes = np.argsort(attributes_sum)
            ordered_x = self.X[ordering_indexes]
            
            pass

        self.old_centroids = np.empty_like(self.centroids)


    def get_labels(self):
        """        
        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        
        distances = distance(self.X, self.centroids)
        self.labels = np.argmin(distances, axis=1)
        self.labels_distances = np.amin(distances, axis=1)


    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        
        self.old_centroids = np.array(self.centroids)
        self.centroids = np.array([np.mean(self.X[self.labels == k], axis=0) for k in range(self.K)])


    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        
        return np.allclose(self.centroids, self.old_centroids, self.options["tolerance"])


    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        
        self.num_iter = 0
        self._init_centroids()

        while not self.converges() and self.num_iter < self.options["max_iter"]:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1


    def whitinClassDistance(self):
        """
         returns the whithin class distance of the current clustering
        """

        # C = self.centroids[self.labels]
        # dist = (self.X - C)
        # return np.mean((dist * dist).sum(1))
        return np.mean(self.labels_distances)


    def silhouette_score(self, K):
        """
         returns the whithin class distance of the current clustering
        """

        # C = self.centroids[self.labels]
        # wcdist = (self.X - C)
        # wcdist = (wcdist * wcdist).sum(1)

        # uff = np.concatenate([self.X[self.labels == k] for k in range(K)])
        # icdist = uff[:,:] - uff[:, :]
        # icdist = (icdist * icdist).sum(1)

        # return np.mean(wcdist / icdist)
        return silhouette_score(self.X, self.labels)
        

    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """

        best_k = self.K
       
        self.K = 2
        self.fit()
        wcd = self.whitinClassDistance()

        for k in range(3, max_K+1):
            self.K = k
            self.fit()
            new_wcd = self.whitinClassDistance()

            if new_wcd/wcd >= 0.8 or k == max_K:
                best_k = k-1
                break

            wcd = new_wcd
        
        self.K = best_k


    def plot_bestK(self, max_K):
        """
         Plot the process of finding the best K
        """
        
        scores = []

        best_k = self.K
       
        self.K = 2
        self.fit()
        score = silhouette_score(self.X, self.labels, metric = 'precomputed')
        scores.append(score)

        for k in range(3, max_K+1):
            self.K = k
            self.fit()
            new_score = silhouette_score(self.X, self.labels, metric = 'precomputed')
            score = new_score
            scores.append(score)
        
        self.K = best_k
        series = pandas.Series(scores, index=list(range(2,max_K+1)))
        series.plot()
        plt.show()





def distance(X, C):
    """
    Calculates the distance between each pixcel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    
    dist = X[:, :, None] - C[:, :, None].T
    return (dist * dist).sum(1)


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
        color_probs: color probabilities
    """
    
    probs = utils.get_color_prob(centroids)
    # probs_max_args = np.argsort(probs.max(1)[::-1])
    # color_probs = probs.max(1)[probs_max_args]
    # probs_argmax = probs.argmax(1)[probs_max_args]
    return utils.colors[probs.argmax(1)]