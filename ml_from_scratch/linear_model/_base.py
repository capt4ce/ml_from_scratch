import numpy as np

class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_columns, n_features = X.shape

        if self.fit_intercept:
            A = np.column_stack((X, np.ones(n_columns)))
        else:
            A = X

        # see /image/linear-regression.png
        beta = np.linalg.inv(A.T @ A) @ A.T @ y

        if self.fit_intercept:
            self.b1 = beta[:-1]
            self.b0 = beta[-1] # intercept
        else:
            self.b1 = beta
            self.b0 = 0

    def predict(self, X):
        X = np.array(X)
        return X @ self.b1 + self.b0