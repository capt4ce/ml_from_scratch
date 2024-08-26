import numpy as np

class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(self, X):
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.random_state:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples//n_splits,dtype=int)
        fold_sizes[:n_samples%n_splits]+=1

        current=0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start: stop]
            current = stop

    def split(self, X):
        # get trainiing set indices
        n_samples = len(X)
        training_indices = np.arange(n_samples)

        for test_index in self._iter_test_indices(X):
            train_index = np.array([idx for idx in training_indices if idx is not test_index])
            yield (train_index, test_index)