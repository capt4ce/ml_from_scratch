from ._base import NearestNeighbors

class KNeighborsClassifier(NearestNeighbors):
    def __init__(self, n_neighbors=5, p=2, weights='uniform'):
        super().__init__(n_neighbors, p)
        self.weights = weights
    
    def predict_proba(self, X):
        indeces= self._kneighbors(X, return_distance=False)

        self._classes = np.unique(self._y[indeces]) 

        proba = np.zeros((X.shape[0], len(self.classes)))

        for i, idx in enumerate(indeces):
            for j, c in enumerate(classes):
                proba[i, j] = np.sum(self._y[idx] == c) / self.n_neighbors
        
        return proba

    def predict(self, X):
        max_proba_idx = np.argmax(self.predict_proba(X), axis=1)
        return self._classes[max_proba_idx]