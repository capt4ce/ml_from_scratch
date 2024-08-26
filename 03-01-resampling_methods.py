import numpy as np
from ml_from_scratch.model_selection import KFold


# CASE SPLITTING 1
# -----------------
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [4, 4]])
y = np.array([1, 2, 3, 4])

kf = KFold(n_splits=2,
           shuffle=True)

for i, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {i}:")
    print(f"    Train: index={train_index}")
    print(f"    Test: index={test_index}")


