import numpy as np

import numpy as np

def _get_distance_weights(dist):
    """
    Get the weights from an array of distances
    Assume weights have already been validated

    Parameters
    ----------
    dist : ndarray
        The input distances
    
    weights : {'uniform', 'distance'}
        The kind of weighting used

    Returns
    -------
    weights_arr : array of the same shape as 'dist'
        If weights=='uniform', then returns None
    """
    return 1.0/(dist**2)

class NearestNeighbors:
    def __init__(self, n_neighbors=5, p=2):
        '''
        Parameters:
        n_neighbors: int, default=5
            Number of neighbors to use for kneighbors queries.
        p: int, default=2
            Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
        '''
        self.n_neighbors = n_neighbors
        self.p = p
    
    def fit(self, X, y):
        '''
        Storing the training input and output data as np arrays
        '''
        self._X = np.array(X)
        self._y = np.array(y)

    def _compute_distances(self, x1, x2):
        '''
        Using Minkowski distance
        '''
        diff = x1 - x2
        abs_diff = np.abs(diff)
        sum_abs_diff = np.sum(np.power(abs_diff, self.p))
        dist = np.power(sum_abs_diff, 1/self.p)
        return dist

    def _kneighbors(self, X, return_distance=True):
        '''
        Finding the k-nearest neighbors for the input data

        Parameters:
        X: array-like of shape (n_samples, n_features)
            The query data.
        return_distance: bool, default=True
            If False, distances will not be returned.
        
        Returns:
        distances: array-like of shape (n_samples, n_neighbors)
            The distances to the nearest neighbors.
        indices: array-like of shape (n_samples, n_neighbors)
            The indices of the nearest neighbors.

        Steps:
        - For each input data, find the distance to all the training data.
        - Sort the distances and get the indices of the k-nearest neighbors.
        - If return_distance is True, return the distances along with the indices.
        '''

        # creating an empty array to store the distances between query data and training data
        # where tA = training data A, and qA = query data A, and dAA = distance between query data A and training data A
        # example of top 3 nearest neighbors
        '''
            tA      tB      tC
        qA  dAA     dBA     dCA
        '''
        distances = np.zeros((X.shape[0], self._X.shape[0]))

        # finding the distance between query data and training data
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                distances[i, j] = self._compute_distances(X[i], self._X[j])

        # sorting data
        indices = np.argsort(distances, axis=1)[:,:self.n_neighbors]

        if return_distance:
            sorted_distances = np.sort(distances, axis=1)[:,:self.n_neighbors]
            return sorted_distances, indices

        return indices

        




