from ._base import NearestNeighbors

class KNeighborsRegressor(NearestNeighbors):
    def __init__(self, n_neighbors=5, p=2, weights='uniform'):
        super().__init__(n_neighbors, p)
        self.weights = weights
    
    def predict(self, X):
        indeces= self._kneighbors(X, return_distance=False)
        neighbors = x[indeces]
        return np.mean(neighbors, axis=1)