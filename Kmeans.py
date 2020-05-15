__authors__ = ['1531206','1456135', '1533031']
__group__ = 'GrupDM12'

import numpy as np
import pandas
import utils
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist
import numpy_indexed as npi


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
            options['fitting'] = 'WCD'
        if 'threshold' not in options:
            options['threshold']  = 0.9

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
        # Funciona bé però pot portar a casos amb empty clusters
        elif self.options['km_init'].lower() == 'sharding':
            attributes_sum = np.sum(self.X, 1)
            ordering_indexes = np.argsort(attributes_sum)
            ordered_x = self.X[ordering_indexes]
            splits = np.array_split(ordered_x, self.K)
            self.centroids = np.empty((self.K, self.X.shape[1]), dtype='float64')

            for p in range(self.K):
                self.centroids[p] = np.mean(splits[p], 0)
        elif self.options['km_init'].lower() == 'kmeans++':
            shape = self.X.shape[0]
            self.centroids = np.empty((self.K, self.X.shape[1]), dtype='float64')
            self.centroids[0] = self.X[np.random.randint(shape), :]

            for k in range(1, self.K):
                distances = np.amin(distance(self.X, self.centroids), axis=1)
                self.centroids[k] = self.X[np.argmax(distances), :] 

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

        for k in range(self.K):
            if np.sum(self.labels == k) != 0:
                self.centroids[k] = np.mean(self.X[self.labels == k], axis=0)

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

    def within_class_distance(self):
        """
         returns the whithin class distance of the current clustering
        """

        clusters = npi.group_by(self.labels).split(self.X)
        sumup = 0.0

        for i in range(self.K):
             sumup += np.mean(euclidean_distances(clusters[i], clusters[i]))

        return sumup

    def within_class_distance_fast(self):
        """
         returns a faster variant of the whithin class distance of the current clustering
        """

        return np.mean(self.labels_distances)

    def inter_class_distance(self):
        """
         returns the inter class distance of the current clustering
        """

        clusters = npi.group_by(self.labels).split(self.X)
        sumup = 0.0

        for i in range(self.K):
            for j in range(i, self.K):
                if i != j:
                    sumup += np.mean(euclidean_distances(clusters[i], clusters[j]))

        return sumup

    def inter_centroids_distance(self):
        """
         returns the inter centroids distance of the current clustering
        """

        return np.mean(euclidean_distances(self.centroids, self.centroids))

    def davis_bouldin_score(self):
        """
         returns the davis bouldin score of the current clustering
        """

        sumup = 0

        for i in range(self.K):
            for j in range(i, self.K):
                if i != j:
                    dist = self.centroids[i] - self.centroids[j]
                    sumup += (self.labels_distances[i] + self.labels_distances[j]) / (dist * dist).sum()
        
        return sumup / self.K

    def silhouette_score(self):
        """
         returns the silhouette score of the current clustering
        """

        a = self.within_class_distance()
        b = self.inter_class_distance()
        return (b-a) / max(a,b)

    def fisher_score(self):
        """
         returns the fisher score of the current clustering
        """
        
        a = self.within_class_distance()
        b = self.inter_class_distance()
        return a/b
        
    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """

        best_k = self.K
       
        self.K = 2
        self.fit()

        if self.options['fitting'] == "WCD":
            score = self.within_class_distance_fast()
        elif self.options['fitting'] == "ICD":
            score = self.inter_class_distance()
        elif self.options['fitting'] == "DB":
            score = self.davis_bouldin_score()
        elif self.options['fitting'] == "fisher":
            score = self.fisher_score()
        elif self.options['fitting'] == "silhouette":
            score = self.silhouette_score()

        for k in range(3, max_K+1):
            self.K = k
            self.fit()

            if self.options['fitting'] == "WCD":
                new_score = self.within_class_distance_fast()
            elif self.options['fitting'] == "ICD":
                new_score = self.inter_class_distance()
            elif self.options['fitting'] == "DB":
                new_score = self.davis_bouldin_score()
            elif self.options['fitting'] == "fisher":
                new_score = self.fisher_score()
            elif self.options['fitting'] == "silhouette":
                new_score = self.silhouette_score()

            if self.options['fitting'] == "silhouette":
                if new_score < score or k == max_K:
                    best_k = k-1
                    break
            elif self.options['fitting'] == "ICD":
                if (new_score/score - 1) < self.options['threshold'] or k == max_K:
                    best_k = k-1
                    break
            else:
                if (1 - new_score/score) < self.options['threshold'] or k == max_K:
                    best_k = k-1
                    break

            score = new_score
        
        self.K = best_k


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