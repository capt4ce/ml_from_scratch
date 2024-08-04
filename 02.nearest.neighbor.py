# Import Library
from ml_from_scratch.neighbors import KNeighborsRegressor
from ml_from_scratch.neighbors import KNeighborsClassifier


# REGRESSION CASE 1
# ----------------
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X,y)

print("Regression Case: ")
print(neigh.predict([[1.5]]))
print("")