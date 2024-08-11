import numpy as np

from ._base import _get_distance_weights
from ._base import NearestNeighbors

class KNeighborsRegressor(NearestNeighbors):
    def __init__(self, n_neighbors=5, p=2, weights='uniform'):
        super().__init__(n_neighbors, p)
        self.weights = weights
    
    def predict(self, X):
        X = np.array(X)

        if self.weights == 'uniform':
            indeces= self._kneighbors(X, return_distance=False)
            return np.mean(self._y[indeces], axis=1)
        else:
            distances, indeces = self._kneighbors(X)
            weights = _get_distance_weights(distances)
            val = np.sum(self._y[indeces] * weights, axis=1)
            denominator = np.sum(weights, axis=1)
            return val/denominator
        
        