# Import Library
from ml_from_scratch.neighbors import KNeighborsRegressor
from ml_from_scratch.neighbors import KNeighborsClassifier


# REGRESSION CASE
# ----------------
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X,y)

print("Regression Case: ")
print(neigh.predict([[1.5],[2.5]]))
print("")

# CLASSIFICATION CASE
# --------------------
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X,y)

print("Classification Case: ")
print(neigh.predict([[1.1]]))
print(neigh.predict_proba([[0.9]]))
print(neigh.classes_)


# REGRESSION CASE - 2
# -------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Prepare data
df = pd.read_csv('data/bmd.csv')
X_train = df['age']
y_train = df['bmd']
X_test = X_train.copy() + 1e-6


# Create prediction
neigh = KNeighborsRegressor(n_neighbors=50)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)


# Plot
fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True, dpi=200)

ax.scatter(X_train, y_train, c="none", edgecolors='grey', label="observed data")
idx_sort = np.argsort(X_test)
X_test_sorted = X_test[idx_sort]
y_pred_sorted = y_pred[idx_sort]
ax.plot(X_test_sorted, y_pred_sorted, c='b', label="prediksi")

ax.legend()
ax.grid(linestyle="--")
ax.set_xlabel("Age")
ax.set_ylabel("BMD")
plt.show()