import numpy as np

from ._base import _get_distance_weights
from ._base import NearestNeighbors

class KNeighborsClassifier(NearestNeighbors):
    def __init__(self, n_neighbors=5, p=2, weights='uniform'):
        super().__init__(n_neighbors, p)
        self.weights = weights

    def predict_proba(self, X):
        X = np.array(X)
        if self.weights == 'uniform':
            indeces = self._kneighbors(X, False)
            weights = None
        else:
            distances, indeces = self._kneighbors(X, True)
            weights = _get_distance_weights(distances)
        sorted_neighbor_outputs = self._y[indeces]

        self.classes_ = np.unique(self._y)

        n_queries = X.shape[0]
        n_classes = len(self.classes_)
        
        neighbors_proba = np.empty((n_queries, n_classes))

        for i in range(n_queries):
            neigh_output = sorted_neighbor_outputs[i]

            for j, class_ in enumerate(self.classes_):
                i_class = (neigh_output == class_).astype(int) #like performing XAND operation ([0 1 0] XAND 0 = [1 0 1])

                if self.weights == 'uniform':
                    class_count = np.sum(i_class)
                else:
                    class_count = np.dot(i_class, weights[i])
                neighbors_proba[i,j] = class_count
        
        for i in range(n_queries):
            sum_i = np.sum(neighbors_proba[i])
            neighbors_proba[i] /= sum_i

        return neighbors_proba
    
    def predict(self, X):
        X = np.array(X)
        proba = self.predict_proba(X)
        max_proba_index = np.argmax(proba)
        return self.classes_[max_proba_index]